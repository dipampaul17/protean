#!/usr/bin/env python3
"""
Clone OSS Airflow/dbt DAGs for Pattern Discovery
Part of Protean Pattern Discovery Engine - Enhanced for Week 1 Mission

GitHub search: 'airflow DAG default_args' 
Clone top 150 repos (shallow --depth 1)
Copy *.py DAGs < 800 LOC into data/raw/dags/
Record repo, sha in CSV manifest.

Runtime <20 min; disk <200 MB.
"""

import os
import sys
import csv
import subprocess
import tempfile
import shutil
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import httpx
import click
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from loguru import logger


@dataclass
class DAGManifestEntry:
    """Manifest entry for cloned DAG"""
    repo_name: str
    repo_url: str
    commit_sha: str
    dag_file: str
    dag_path: str
    lines_of_code: int
    contains_default_args: bool
    contains_schedule: bool
    timestamp: str


class DagCloner:
    """Efficient OSS DAG cloner for pattern discovery"""
    
    def __init__(self, github_token: str, output_dir: str = "data/raw/oss_dags", max_repos: int = 150):
        self.github_token = github_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_repos = max_repos
        self.max_loc = 800  # Lines of code limit
        self.cloned_repos = []
        self.dag_manifest = []
        
        # Enhanced search patterns for high-quality DAGs
        self.search_queries = [
            # Core Airflow patterns
            "airflow DAG default_args language:Python",
            "from airflow import DAG language:Python", 
            "airflow.operators language:Python",
            "default_args dag_id language:Python",
            
            # dbt patterns  
            "dbt airflow language:Python",
            "dbt_dag language:Python",
            
            # Production patterns
            "airflow production language:Python",
            "airflow etl language:Python", 
            "airflow pipeline language:Python",
            
            # Data engineering patterns
            "airflow data engineering language:Python",
            "airflow workflow language:Python"
        ]
        
        # DAG quality indicators
        self.dag_indicators = [
            "from airflow import DAG",
            "from airflow.models import DAG", 
            "default_args",
            "dag_id",
            "schedule_interval",
            "start_date",
            "@dag",  # Airflow 2.0+ decorator
            "with DAG"
        ]
        
        # High-value operator patterns
        self.operator_patterns = [
            "BashOperator", "PythonOperator", "SqlOperator", "PostgresOperator",
            "S3KeySensor", "DockerOperator", "KubernetesOperator", 
            "BigQueryOperator", "DataprocOperator", "SparkSubmitOperator",
            "dbt", "DataFlowPythonOperator", "EMRAddStepsOperator"
        ]
    
    async def search_github_repos(self) -> List[Dict[str, Any]]:
        """Search GitHub for repositories with Airflow DAGs"""
        repos = []
        seen_repos = set()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            for query in self.search_queries:
                try:
                    logger.info(f"🔍 Searching: {query}")
                    
                    params = {
                        "q": f"{query} stars:>5",  # Filter for quality repos
                        "sort": "stars",
                        "order": "desc",
                        "per_page": 30,
                        "page": 1
                    }
                    
                    response = await client.get("https://api.github.com/search/repositories", 
                                              headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    for repo in data.get("items", []):
                        repo_name = repo["full_name"]
                        if repo_name not in seen_repos and len(repos) < self.max_repos:
                            repos.append(repo)
                            seen_repos.add(repo_name)
                            logger.debug(f"Found repo: {repo_name} (⭐{repo['stargazers_count']})")
                    
                    # Rate limiting
                    await asyncio.sleep(1.0)
                    
                    if len(repos) >= self.max_repos:
                        break
                        
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
                    continue
        
        logger.info(f"✅ Found {len(repos)} unique repositories")
        return repos[:self.max_repos]
    
    def count_lines_of_code(self, file_path: Path) -> int:
        """Count non-empty, non-comment lines of code"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            loc = 0
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    loc += 1
            
            return loc
        except Exception:
            return 0
    
    def is_high_quality_dag(self, file_path: Path) -> Dict[str, Any]:
        """Check if file is a high-quality Airflow DAG"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Must contain core DAG indicators
            dag_score = sum(1 for indicator in self.dag_indicators if indicator in content)
            if dag_score < 2:  # Need at least 2 DAG indicators
                return {"is_dag": False}
            
            # Check for production-quality patterns
            has_default_args = "default_args" in content
            has_schedule = any(pattern in content for pattern in ["schedule_interval", "schedule", "@daily", "@hourly"])
            has_operators = sum(1 for op in self.operator_patterns if op in content)
            
            # Quality scoring
            quality_score = dag_score + has_operators + (2 if has_default_args else 0)
            
            return {
                "is_dag": quality_score >= 3,  # Minimum quality threshold
                "contains_default_args": has_default_args,
                "contains_schedule": has_schedule,
                "operator_count": has_operators,
                "quality_score": quality_score
            }
            
        except Exception:
            return {"is_dag": False}
    
    def clone_repo_shallow(self, repo_url: str, repo_name: str, temp_dir: Path) -> Optional[str]:
        """Shallow clone repository and return commit SHA"""
        try:
            clone_path = temp_dir / repo_name.replace("/", "_")
            
            # Shallow clone for efficiency
            cmd = [
                "git", "clone", 
                "--depth", "1", 
                "--single-branch",
                repo_url, 
                str(clone_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.warning(f"Failed to clone {repo_name}: {result.stderr}")
                return None
            
            # Get commit SHA
            sha_cmd = ["git", "-C", str(clone_path), "rev-parse", "HEAD"]
            sha_result = subprocess.run(sha_cmd, capture_output=True, text=True)
            commit_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else "unknown"
            
            return commit_sha
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Clone timeout for {repo_name}")
            return None
        except Exception as e:
            logger.warning(f"Clone error for {repo_name}: {e}")
            return None
    
    def extract_dags_from_repo(self, repo_path: Path, repo_name: str, commit_sha: str) -> List[DAGManifestEntry]:
        """Extract high-quality DAG files from cloned repository"""
        dags = []
        
        # Search for Python files recursively
        python_files = list(repo_path.rglob("*.py"))
        
        for py_file in python_files:
            try:
                # Skip test files and common non-DAG patterns
                if any(pattern in str(py_file).lower() for pattern in 
                       ["test", "__pycache__", ".git", "setup.py", "conftest"]):
                    continue
                
                # Check line count
                loc = self.count_lines_of_code(py_file)
                if loc == 0 or loc > self.max_loc:
                    continue
                
                # Check if it's a high-quality DAG
                dag_info = self.is_high_quality_dag(py_file)
                if not dag_info.get("is_dag", False):
                    continue
                
                # Create unique filename
                relative_path = py_file.relative_to(repo_path)
                safe_filename = f"{repo_name.replace('/', '_')}_{relative_path.as_posix().replace('/', '_')}"
                
                # Copy to output directory
                output_path = self.output_dir / safe_filename
                shutil.copy2(py_file, output_path)
                
                # Create manifest entry
                manifest_entry = DAGManifestEntry(
                    repo_name=repo_name,
                    repo_url=f"https://github.com/{repo_name}",
                    commit_sha=commit_sha,
                    dag_file=safe_filename,
                    dag_path=str(relative_path),
                    lines_of_code=loc,
                    contains_default_args=dag_info.get("contains_default_args", False),
                    contains_schedule=dag_info.get("contains_schedule", False),
                    timestamp=datetime.now().isoformat()
                )
                
                dags.append(manifest_entry)
                logger.debug(f"Extracted DAG: {safe_filename} ({loc} LOC)")
                
            except Exception as e:
                logger.warning(f"Error processing {py_file}: {e}")
                continue
        
        return dags
    
    def save_manifest(self, filename: str = "oss_dags_manifest.csv") -> str:
        """Save DAG manifest to CSV"""
        manifest_path = self.output_dir / filename
        
        with open(manifest_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'repo_name', 'repo_url', 'commit_sha', 'dag_file', 'dag_path',
                'lines_of_code', 'contains_default_args', 'contains_schedule', 'timestamp'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in self.dag_manifest:
                writer.writerow({
                    'repo_name': entry.repo_name,
                    'repo_url': entry.repo_url,
                    'commit_sha': entry.commit_sha,
                    'dag_file': entry.dag_file,
                    'dag_path': entry.dag_path,
                    'lines_of_code': entry.lines_of_code,
                    'contains_default_args': entry.contains_default_args,
                    'contains_schedule': entry.contains_schedule,
                    'timestamp': entry.timestamp
                })
        
        logger.info(f"📄 Saved manifest to {manifest_path}")
        return str(manifest_path)
    
    async def clone_oss_dags(self) -> Dict[str, Any]:
        """Main function to clone OSS DAGs"""
        start_time = datetime.now()
        logger.info(f"🚀 Starting OSS DAG cloning (target: {self.max_repos} repos)")
        
        # Search for repositories
        repos = await self.search_github_repos()
        if not repos:
            logger.error("❌ No repositories found")
            return {"success": False, "dags_count": 0}
        
        # Create temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Clone repositories and extract DAGs
            for repo in tqdm(repos, desc="Cloning repos"):
                repo_name = repo["full_name"]
                repo_url = repo["clone_url"]
                
                try:
                    # Clone repository
                    commit_sha = self.clone_repo_shallow(repo_url, repo_name, temp_path)
                    if not commit_sha:
                        continue
                    
                    self.cloned_repos.append(repo_name)
                    
                    # Extract DAGs
                    repo_path = temp_path / repo_name.replace("/", "_")
                    if repo_path.exists():
                        dags = self.extract_dags_from_repo(repo_path, repo_name, commit_sha)
                        self.dag_manifest.extend(dags)
                        
                        logger.debug(f"✅ {repo_name}: {len(dags)} DAGs extracted")
                    
                    # Cleanup repo to save disk space
                    if repo_path.exists():
                        shutil.rmtree(repo_path, ignore_errors=True)
                    
                except Exception as e:
                    logger.warning(f"❌ Failed to process {repo_name}: {e}")
                    continue
        
        # Save manifest
        manifest_path = self.save_manifest()
        
        # Calculate statistics
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        # Check disk usage
        total_size = sum(f.stat().st_size for f in self.output_dir.glob("*.py")) / (1024 * 1024)  # MB
        
        stats = {
            "success": True,
            "repos_searched": len(repos),
            "repos_cloned": len(self.cloned_repos),
            "dags_extracted": len(self.dag_manifest),
            "runtime_seconds": runtime,
            "disk_usage_mb": total_size,
            "manifest_path": manifest_path,
            "avg_loc": sum(dag.lines_of_code for dag in self.dag_manifest) / len(self.dag_manifest) if self.dag_manifest else 0,
            "with_default_args": sum(1 for dag in self.dag_manifest if dag.contains_default_args),
            "with_schedule": sum(1 for dag in self.dag_manifest if dag.contains_schedule)
        }
        
        logger.info("📊 OSS DAG Cloning Complete:")
        logger.info(f"   Repos cloned: {stats['repos_cloned']}")
        logger.info(f"   DAGs extracted: {stats['dags_extracted']}")
        logger.info(f"   Runtime: {stats['runtime_seconds']:.1f}s")
        logger.info(f"   Disk usage: {stats['disk_usage_mb']:.1f} MB")
        logger.info(f"   Avg LOC: {stats['avg_loc']:.0f}")
        logger.info(f"   With default_args: {stats['with_default_args']}")
        logger.info(f"   With schedules: {stats['with_schedule']}")
        
        return stats


@click.command()
@click.option('--max-repos', default=150, type=int, help='Maximum repositories to clone')
@click.option('--output-dir', default="data/raw/oss_dags", help='Output directory for DAGs')
def main(max_repos: int, output_dir: str):
    """Clone OSS Airflow/dbt DAGs for pattern discovery"""
    asyncio.run(run_clone(max_repos, output_dir))


async def run_clone(max_repos: int, output_dir: str):
    """Async wrapper for cloning"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.error("❌ GITHUB_TOKEN not found in environment variables")
        return
    
    # Initialize cloner
    cloner = DagCloner(github_token, output_dir, max_repos)
    
    try:
        stats = await cloner.clone_oss_dags()
        
        if stats["success"]:
            logger.info("✅ OSS DAG cloning completed successfully")
            
            # Verify constraints
            if stats["runtime_seconds"] > 1200:  # 20 minutes
                logger.warning(f"⚠️  Runtime exceeded 20min: {stats['runtime_seconds']:.1f}s")
            
            if stats["disk_usage_mb"] > 200:  # 200 MB
                logger.warning(f"⚠️  Disk usage exceeded 200MB: {stats['disk_usage_mb']:.1f}MB")
            else:
                logger.info(f"✅ Constraints met: {stats['runtime_seconds']:.1f}s, {stats['disk_usage_mb']:.1f}MB")
        else:
            logger.error("❌ OSS DAG cloning failed")
            
    except Exception as e:
        logger.error(f"❌ Clone failed: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Run with defaults
        asyncio.run(run_clone(150, "data/raw/oss_dags"))
    else:
        # Use CLI
        main()
