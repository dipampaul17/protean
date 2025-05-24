#!/usr/bin/env python3
"""
Build Corpus Index for Pattern Discovery
Part of Protean Pattern Discovery Engine - Week 1 Mission

Walk data/raw/{postmortems,stackoverflow,oss_dags}/
• For each doc: simple heuristic tag → {'pattern_hint': 'CircuitBreaker'} 
• Store to sqlite `corpus.db` table(doc_id, source, text, pattern_hint)
• For DAG .py files: Parse NetworkX graph (operators as nodes)
• Fingerprint op types → {'dag_id':..., 'ops':['PythonOperator', 'S3ToRedshiftOperator'...]}
• Store to table(dag_id, repo, json_graph)

Runtime target: <40 min
"""

import os
import sys
import json
import sqlite3
import hashlib
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import gzip

import pandas as pd
import networkx as nx
from loguru import logger
from tqdm import tqdm


@dataclass
class Document:
    """Processed document for corpus indexing"""
    doc_id: str
    source: str
    text: str
    pattern_hint: Optional[str]
    metadata: Dict[str, Any]
    created_at: str


@dataclass 
class DAGGraph:
    """Parsed DAG graph structure"""
    dag_id: str
    repo: str
    operators: List[str]
    graph_json: str
    complexity_score: int
    metadata: Dict[str, Any]
    created_at: str


class PatternHinter:
    """Heuristic pattern detection for corpus documents"""
    
    def __init__(self):
        # Enhanced failure patterns from postmortem harvester
        self.failure_patterns = {
            "CircuitBreaker": [
                "circuit breaker", "circuit break", "circuit open", "fallback", 
                "retry", "timeout", "bulkhead", "hystrix", "resilience"
            ],
            "CascadingFailure": [
                "cascade", "cascading", "domino effect", "chain reaction", 
                "ripple effect", "snowball", "amplification", "downstream"
            ],
            "ResourceExhaustion": [
                "memory leak", "cpu spike", "disk space", "out of memory", "oom",
                "resource exhaustion", "capacity", "throttling", "rate limit",
                "connection pool", "thread pool", "queue full"
            ],
            "NetworkPartition": [
                "network partition", "split brain", "network connectivity", 
                "dns failure", "timeout", "latency spike", "packet loss",
                "network isolation", "partition tolerance"
            ],
            "ConfigurationDrift": [
                "configuration", "config drift", "misconfiguration", 
                "deployment", "rollback", "feature flag", "canary",
                "environment mismatch", "config error"
            ],
            "DependencyFailure": [
                "upstream", "downstream", "third party", "external service",
                "vendor", "api failure", "service unavailable", "dependency"
            ],
            "DataCorruption": [
                "data corruption", "data loss", "inconsistent state", 
                "transaction", "rollback", "backup", "recovery",
                "data integrity", "corruption"
            ],
            "ScalingFailure": [
                "auto scaling", "horizontal scaling", "vertical scaling", 
                "load balancer", "traffic spike", "capacity planning",
                "scale out", "scale up", "elasticity"
            ],
            "MonitoringBlindSpot": [
                "monitoring", "alerting", "observability", "metrics", 
                "no visibility", "silent failure", "detection delay",
                "blind spot", "monitoring gap"
            ],
            "DeploymentFailure": [
                "deployment", "deploy", "release", "rollout", "blue green",
                "canary deployment", "rolling update", "deployment pipeline"
            ],
            "DatabaseFailure": [
                "database", "db", "sql", "nosql", "deadlock", "lock contention",
                "connection timeout", "query performance", "index"
            ],
            "LoadBalancingIssue": [
                "load balancer", "load balancing", "traffic distribution",
                "sticky sessions", "health check", "upstream"
            ]
        }
        
        # Infrastructure patterns for DAGs and stack overflow
        self.infrastructure_patterns = {
            "ETLPipeline": [
                "etl", "extract transform load", "data pipeline", "data processing",
                "batch processing", "stream processing"
            ],
            "MLOpsWorkflow": [
                "mlops", "machine learning", "model training", "feature engineering",
                "model deployment", "ml pipeline"
            ],
            "DataWarehouse": [
                "data warehouse", "dwh", "olap", "dimensional modeling",
                "star schema", "snowflake schema", "fact table"
            ],
            "Microservices": [
                "microservices", "service mesh", "api gateway", "service discovery",
                "distributed system", "microservice"
            ],
            "CloudNative": [
                "cloud native", "kubernetes", "containerization", "serverless",
                "faas", "container orchestration"
            ],
            "EventDriven": [
                "event driven", "event sourcing", "message queue", "pub sub",
                "kafka", "rabbitmq", "event streaming"
            ]
        }
    
    def detect_pattern(self, text: str) -> Optional[str]:
        """Detect primary pattern in text using heuristics"""
        text_lower = text.lower()
        
        # Score each pattern
        pattern_scores = {}
        
        # Check failure patterns first (higher priority)
        for pattern, keywords in self.failure_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                pattern_scores[pattern] = score * 2  # Higher weight for failure patterns
        
        # Check infrastructure patterns
        for pattern, keywords in self.infrastructure_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                pattern_scores[pattern] = score
        
        # Return highest scoring pattern
        if pattern_scores:
            return max(pattern_scores, key=pattern_scores.get)
        
        return None


class DAGParser:
    """Parse Airflow DAG files into NetworkX graphs"""
    
    def __init__(self):
        # Common Airflow operators to identify
        self.airflow_operators = {
            # Core operators
            "BashOperator", "PythonOperator", "EmailOperator", "DummyOperator", "EmptyOperator",
            # Database operators
            "SqlOperator", "PostgresOperator", "MySqlOperator", "SqliteOperator",
            # Cloud operators
            "S3KeySensor", "S3ToRedshiftOperator", "RedshiftToS3Operator",
            "BigQueryOperator", "DataprocOperator", "DataFlowPythonOperator",
            "EMRAddStepsOperator", "EMRCreateJobFlowOperator",
            # Container operators
            "DockerOperator", "KubernetesOperator", "KubernetesPodOperator",
            # Data operators
            "SparkSubmitOperator", "HiveOperator", "PrestoOperator",
            # dbt operators
            "DbtRunOperator", "DbtTestOperator", "DbtSeedOperator",
            # Sensors
            "FileSensor", "S3KeySensor", "TimeSensor", "ExternalTaskSensor",
            # Transfer operators
            "S3ToHiveOperator", "HiveToMySqlOperator", "MySqlToHiveOperator"
        }
    
    def extract_dag_id(self, content: str) -> Optional[str]:
        """Extract DAG ID from content"""
        # Look for dag_id in DAG definition
        patterns = [
            r'dag_id\s*=\s*["\']([^"\']+)["\']',
            r'DAG\s*\(\s*["\']([^"\']+)["\']',
            r'@dag\s*\(\s*dag_id\s*=\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        return None
    
    def extract_operators(self, content: str) -> List[str]:
        """Extract operator types from DAG content"""
        operators = []
        
        # Look for operator instantiations
        for operator in self.airflow_operators:
            if operator in content:
                operators.append(operator)
        
        # Also look for custom operators
        operator_pattern = r'(\w+Operator)\s*\('
        matches = re.findall(operator_pattern, content)
        for match in matches:
            if match not in operators and match.endswith('Operator'):
                operators.append(match)
        
        return list(set(operators))  # Remove duplicates
    
    def parse_dag_structure(self, content: str) -> nx.DiGraph:
        """Parse DAG structure into NetworkX graph"""
        graph = nx.DiGraph()
        
        try:
            # Try to parse the AST
            tree = ast.parse(content)
            
            # Look for task definitions and dependencies
            task_ids = []
            dependencies = []
            
            for node in ast.walk(tree):
                # Look for task instantiations
                if isinstance(node, ast.Assign):
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        task_name = node.targets[0].id
                        if isinstance(node.value, ast.Call) and hasattr(node.value.func, 'id'):
                            if any(op in node.value.func.id for op in ['Operator', 'Sensor']):
                                task_ids.append(task_name)
                                graph.add_node(task_name, type=node.value.func.id)
                
                # Look for dependencies (>> operator)
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
                    # This is a simplified approach - real DAG parsing would be more complex
                    pass
            
            # Add nodes if we found tasks
            if task_ids:
                for i, task_id in enumerate(task_ids):
                    if not graph.has_node(task_id):
                        graph.add_node(task_id, type="UnknownOperator")
                    
                    # Add sequential dependencies as a fallback
                    if i > 0:
                        graph.add_edge(task_ids[i-1], task_id)
        
        except Exception as e:
            logger.debug(f"Failed to parse DAG AST: {e}")
            # Fallback: create simple graph from operators
            operators = self.extract_operators(content)
            for i, op in enumerate(operators):
                graph.add_node(f"task_{i}", type=op)
                if i > 0:
                    graph.add_edge(f"task_{i-1}", f"task_{i}")
        
        return graph
    
    def calculate_complexity(self, graph: nx.DiGraph, operators: List[str]) -> int:
        """Calculate DAG complexity score"""
        score = 0
        
        # Node count
        score += len(graph.nodes) * 2
        
        # Edge count (dependencies)
        score += len(graph.edges) * 3
        
        # Operator diversity
        score += len(set(operators)) * 5
        
        # Special high-value operators
        high_value_ops = ["KubernetesOperator", "SparkSubmitOperator", "BigQueryOperator"]
        score += sum(5 for op in operators if any(hv in op for hv in high_value_ops))
        
        return min(score, 100)  # Cap at 100


class CorpusIndexer:
    """Main corpus indexing engine"""
    
    def __init__(self, db_path: str = "corpus.db"):
        self.db_path = db_path
        self.pattern_hinter = PatternHinter()
        self.dag_parser = DAGParser()
        self.processed_docs = 0
        self.processed_dags = 0
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                pattern_hint TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Create indexes separately
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_pattern_hint ON documents(pattern_hint)')
        
        # DAGs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dags (
                dag_id TEXT PRIMARY KEY,
                repo TEXT NOT NULL,
                operators TEXT NOT NULL,
                graph_json TEXT NOT NULL,
                complexity_score INTEGER,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Create indexes separately  
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dags_repo ON dags(repo)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dags_complexity ON dags(complexity_score)')
        
        # Pattern hints summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_summary (
                pattern_name TEXT PRIMARY KEY,
                document_count INTEGER,
                dag_count INTEGER,
                total_count INTEGER,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Initialized database: {self.db_path}")
    
    def generate_doc_id(self, source: str, content: str) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"{source}_{content_hash}"
    
    def process_json_file(self, file_path: Path, source: str) -> List[Document]:
        """Process JSON file containing documents"""
        documents = []
        
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and 'items' in data:
                items = data['items']
            else:
                items = [data]
            
            for item in items:
                # Extract text content
                text_content = ""
                if isinstance(item, dict):
                    # Try different content fields
                    for field in ['content', 'body', 'question', 'answer', 'text', 'title']:
                        if field in item and item[field]:
                            text_content += str(item[field]) + " "
                    
                    # For Stack Overflow, combine question and answers
                    if 'answers' in item and item['answers']:
                        for answer in item['answers']:
                            if 'body' in answer:
                                text_content += str(answer['body']) + " "
                else:
                    text_content = str(item)
                
                if len(text_content.strip()) < 100:  # Skip very short content
                    continue
                
                # Generate document
                doc_id = self.generate_doc_id(source, text_content)
                pattern_hint = self.pattern_hinter.detect_pattern(text_content)
                
                doc = Document(
                    doc_id=doc_id,
                    source=source,
                    text=text_content.strip(),
                    pattern_hint=pattern_hint,
                    metadata=item if isinstance(item, dict) else {},
                    created_at=datetime.now().isoformat()
                )
                documents.append(doc)
        
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
        
        return documents
    
    def process_parquet_file(self, file_path: Path, source: str) -> List[Document]:
        """Process Parquet file containing documents"""
        documents = []
        
        try:
            df = pd.read_parquet(file_path)
            
            for _, row in df.iterrows():
                # Extract text content
                text_content = ""
                for field in ['content', 'body', 'question', 'answer', 'text', 'title']:
                    if field in row and pd.notna(row[field]):
                        text_content += str(row[field]) + " "
                
                if len(text_content.strip()) < 100:  # Skip very short content
                    continue
                
                # Generate document
                doc_id = self.generate_doc_id(source, text_content)
                pattern_hint = self.pattern_hinter.detect_pattern(text_content)
                
                doc = Document(
                    doc_id=doc_id,
                    source=source,
                    text=text_content.strip(),
                    pattern_hint=pattern_hint,
                    metadata=row.to_dict(),
                    created_at=datetime.now().isoformat()
                )
                documents.append(doc)
        
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
        
        return documents
    
    def process_dag_file(self, file_path: Path) -> Optional[DAGGraph]:
        """Process DAG Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract DAG information
            dag_id = self.dag_parser.extract_dag_id(content)
            if not dag_id:
                dag_id = file_path.stem
            
            operators = self.dag_parser.extract_operators(content)
            if not operators:  # Skip files without operators
                return None
            
            # Parse graph structure
            graph = self.dag_parser.parse_dag_structure(content)
            graph_json = json.dumps(nx.node_link_data(graph))
            
            # Calculate complexity
            complexity = self.dag_parser.calculate_complexity(graph, operators)
            
            # Extract repo from path
            repo = "unknown"
            path_parts = file_path.parts
            for part in path_parts:
                if "_" in part and len(part) > 10:
                    repo = part.split("_")[0] + "/" + part.split("_")[1]
                    break
            
            dag_graph = DAGGraph(
                dag_id=dag_id,
                repo=repo,
                operators=operators,
                graph_json=graph_json,
                complexity_score=complexity,
                metadata={
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "lines_of_code": len(content.split('\n'))
                },
                created_at=datetime.now().isoformat()
            )
            
            return dag_graph
        
        except Exception as e:
            logger.warning(f"Failed to process DAG {file_path}: {e}")
            return None
    
    def save_documents(self, documents: List[Document]):
        """Save documents to database"""
        if not documents:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for doc in documents:
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (doc_id, source, text, pattern_hint, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                doc.doc_id, doc.source, doc.text, doc.pattern_hint,
                json.dumps(doc.metadata), doc.created_at
            ))
        
        conn.commit()
        conn.close()
        self.processed_docs += len(documents)
    
    def save_dags(self, dags: List[DAGGraph]):
        """Save DAGs to database"""
        if not dags:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for dag in dags:
            cursor.execute('''
                INSERT OR REPLACE INTO dags 
                (dag_id, repo, operators, graph_json, complexity_score, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                dag.dag_id, dag.repo, json.dumps(dag.operators), dag.graph_json,
                dag.complexity_score, json.dumps(dag.metadata), dag.created_at
            ))
        
        conn.commit()
        conn.close()
        self.processed_dags += len(dags)
    
    def update_pattern_summary(self):
        """Update pattern summary statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing summary
        cursor.execute('DELETE FROM pattern_summary')
        
        # Document patterns
        cursor.execute('''
            SELECT pattern_hint, COUNT(*) as count 
            FROM documents 
            WHERE pattern_hint IS NOT NULL 
            GROUP BY pattern_hint
        ''')
        doc_patterns = dict(cursor.fetchall())
        
        # DAG patterns (based on operators)
        cursor.execute('SELECT operators FROM dags')
        dag_operators = cursor.fetchall()
        
        dag_patterns = {}
        for (ops_json,) in dag_operators:
            ops = json.loads(ops_json)
            for op in ops:
                # Map operators to patterns
                if any(pattern in op for pattern in ['Kubernetes', 'Docker']):
                    dag_patterns['CloudNative'] = dag_patterns.get('CloudNative', 0) + 1
                elif any(pattern in op for pattern in ['BigQuery', 'Dataproc', 'EMR', 'Spark']):
                    dag_patterns['ETLPipeline'] = dag_patterns.get('ETLPipeline', 0) + 1
                elif 'dbt' in op.lower():
                    dag_patterns['DataWarehouse'] = dag_patterns.get('DataWarehouse', 0) + 1
        
        # Combine and save
        all_patterns = set(doc_patterns.keys()) | set(dag_patterns.keys())
        for pattern in all_patterns:
            doc_count = doc_patterns.get(pattern, 0)
            dag_count = dag_patterns.get(pattern, 0)
            total_count = doc_count + dag_count
            
            cursor.execute('''
                INSERT INTO pattern_summary 
                (pattern_name, document_count, dag_count, total_count, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (pattern, doc_count, dag_count, total_count, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def index_corpus(self, data_dir: str = "data/raw"):
        """Main corpus indexing function"""
        start_time = datetime.now()
        logger.info("🚀 Starting corpus indexing for pattern discovery")
        
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"❌ Data directory not found: {data_dir}")
            return
        
        # Process postmortems
        postmortem_dir = data_path / "postmortems"
        if postmortem_dir.exists():
            logger.info("📄 Processing postmortem documents...")
            for file_path in postmortem_dir.glob("*.json"):
                docs = self.process_json_file(file_path, "postmortem")
                self.save_documents(docs)
            
            for file_path in postmortem_dir.glob("*.parquet"):
                docs = self.process_parquet_file(file_path, "postmortem")
                self.save_documents(docs)
        
        # Process Stack Overflow
        stackoverflow_dir = data_path / "stackoverflow"
        if stackoverflow_dir.exists():
            logger.info("📄 Processing Stack Overflow documents...")
            for file_path in stackoverflow_dir.glob("*.json.gz"):
                docs = self.process_json_file(file_path, "stackoverflow")
                self.save_documents(docs)
            
            for file_path in stackoverflow_dir.glob("*.parquet"):
                docs = self.process_parquet_file(file_path, "stackoverflow")
                self.save_documents(docs)
        
        # Process OSS DAGs
        dags_dir = data_path / "oss_dags"
        if dags_dir.exists():
            logger.info("🔧 Processing OSS DAG files...")
            dag_files = list(dags_dir.glob("*.py"))
            
            for file_path in tqdm(dag_files, desc="Processing DAGs"):
                dag_graph = self.process_dag_file(file_path)
                if dag_graph:
                    self.save_dags([dag_graph])
        
        # Update pattern summary
        logger.info("📊 Updating pattern summary...")
        self.update_pattern_summary()
        
        # Generate final statistics
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM documents')
        total_docs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM dags')
        total_dags = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM pattern_summary')
        total_patterns = cursor.fetchone()[0]
        
        conn.close()
        
        logger.info("✅ Corpus indexing complete!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Documents indexed: {total_docs}")
        logger.info(f"   DAGs indexed: {total_dags}")
        logger.info(f"   Patterns identified: {total_patterns}")
        logger.info(f"   Runtime: {runtime:.1f}s")
        logger.info(f"   Database: {self.db_path}")
        
        return {
            "success": True,
            "total_documents": total_docs,
            "total_dags": total_dags,
            "total_patterns": total_patterns,
            "runtime_seconds": runtime,
            "database_path": self.db_path
        }


def main():
    """Main execution function"""
    try:
        indexer = CorpusIndexer()
        stats = indexer.index_corpus()
        
        if stats["success"]:
            logger.info("🎯 Corpus ready for pattern discovery!")
        else:
            logger.error("❌ Corpus indexing failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️  Indexing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Indexing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 