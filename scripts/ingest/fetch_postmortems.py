#!/usr/bin/env python3
"""
Fetch Infrastructure Postmortems for Pattern Analysis
Part of Protean Pattern Discovery Engine - Enhanced for Week 1 Mission
"""

import os
import json
import time
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import httpx
import pandas as pd
import click
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from loguru import logger


class SeenIndex:
    """Track seen URLs and content hashes to avoid duplicates"""
    
    def __init__(self):
        self.seen_urls: Set[str] = set()
        self.content_hashes: Set[int] = set()
    
    def is_duplicate(self, url: str, content: str) -> bool:
        """Check if URL or content has been seen before"""
        if url in self.seen_urls:
            return True
        
        # Use first 500 chars for content similarity
        content_hash = hash(content[:500]) if content else 0
        if content_hash in self.content_hashes:
            return True
        
        return False
    
    def add(self, url: str, content: str):
        """Add URL and content to seen index"""
        self.seen_urls.add(url)
        if content:
            content_hash = hash(content[:500])
            self.content_hashes.add(content_hash)
    
    def stats(self) -> Dict[str, int]:
        """Get statistics"""
        return {
            "seen_urls": len(self.seen_urls),
            "content_hashes": len(self.content_hashes)
        }


@dataclass
class PostmortemData:
    """Enhanced structure for postmortem incident data with pattern analysis focus"""
    source: str
    url: str
    title: str
    content: str
    timestamp: str
    tags: List[str]
    severity: Optional[str] = None
    services_affected: List[str] = None
    root_cause: Optional[str] = None
    resolution_time: Optional[int] = None  # minutes
    infrastructure_components: List[str] = None
    failure_pattern: Optional[str] = None  # NEW: Pattern classification
    timeline_events: List[Dict[str, str]] = None  # NEW: Timeline extraction
    blast_radius: Optional[str] = None  # NEW: Impact scope
    detection_time: Optional[int] = None  # NEW: Time to detect
    mitigation_actions: List[str] = None  # NEW: What was done to fix
    quality_score: float = 0.0  # NEW: Content quality for pattern analysis


class PostmortemHarvester:
    """Enhanced harvester for infrastructure postmortems with pattern focus"""
    
    def __init__(self, github_token: str, output_dir: str = "data/raw/postmortems"):
        self.github_token = github_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seen_index = SeenIndex()  # Track duplicates
        
        # Enhanced postmortem repositories - high quality sources
        self.github_repos = [
            # Core postmortem collections
            "danluu/post-mortems",
            "upgundecha/howtheyfailed",
            "kahun/awesome-sysadmin",
            
            # Company engineering blogs with frequent postmortems
            "github/backup-utils",  # GitHub's infrastructure
            "kubernetes/kubernetes",  # K8s issues often have postmortem-style content
            "istio/istio",  # Service mesh failures
            "prometheus/prometheus",  # Monitoring system issues
            "grafana/grafana",  # Observability platform issues
            
            # SRE and DevOps focused repos
            "google/sre-book",  # SRE practices and case studies
            "spotify/backstage",  # Platform engineering incidents
            "hashicorp/terraform",  # Infrastructure provisioning issues
            "elastic/elasticsearch",  # Database reliability issues
            "apache/kafka",  # Streaming platform incidents
            
            # Infrastructure failure databases
            "mtdvio/every-programmer-should-know",
            "donnemartin/system-design-primer",
            "awesome-foss/awesome-sysadmin",
            "avelino/awesome-go",  # Go infrastructure tools
            "sindresorhus/awesome",  # General awesome lists may contain postmortems
            
            # Monitoring and observability repos (often have incident analysis)
            "grafana/loki",
            "jaegertracing/jaeger",
            "open-telemetry/opentelemetry-specification"
        ]
        
        # Precise failure pattern categories for pattern discovery
        self.failure_patterns = {
            "cascading_failure": [
                "cascade", "cascading", "domino effect", "chain reaction", 
                "ripple effect", "snowball", "amplification"
            ],
            "resource_exhaustion": [
                "memory leak", "cpu spike", "disk space", "out of memory", "oom",
                "resource exhaustion", "capacity", "throttling", "rate limit"
            ],
            "network_partition": [
                "network partition", "split brain", "network connectivity", 
                "dns failure", "timeout", "latency spike", "packet loss"
            ],
            "configuration_drift": [
                "configuration", "config drift", "misconfiguration", 
                "deployment", "rollback", "feature flag", "canary"
            ],
            "dependency_failure": [
                "upstream", "downstream", "third party", "external service",
                "vendor", "api failure", "service unavailable"
            ],
            "data_corruption": [
                "data corruption", "data loss", "inconsistent state", 
                "transaction", "rollback", "backup", "recovery"
            ],
            "scaling_failure": [
                "auto scaling", "horizontal scaling", "vertical scaling", 
                "load balancer", "traffic spike", "capacity planning"
            ],
            "monitoring_blind_spot": [
                "monitoring", "alerting", "observability", "metrics", 
                "no visibility", "silent failure", "detection delay"
            ]
        }
        
        # Enhanced infrastructure keywords with pattern implications
        self.infra_keywords = {
            # Container orchestration
            "kubernetes": ["k8s", "pod", "deployment", "service mesh", "ingress"],
            "docker": ["container", "containerization", "dockerfile", "registry"],
            
            # Cloud platforms
            "aws": ["ec2", "s3", "rds", "lambda", "cloudformation", "ecs", "eks"],
            "gcp": ["gke", "compute engine", "cloud sql", "pub/sub"],
            "azure": ["vm", "blob storage", "cosmos db", "aks"],
            
            # Databases and storage
            "postgresql": ["postgres", "pg", "database", "sql"],
            "redis": ["cache", "session store", "pub/sub"],
            "elasticsearch": ["search", "indexing", "cluster"],
            
            # Message queues and streaming
            "kafka": ["streaming", "topic", "partition", "consumer lag"],
            "rabbitmq": ["message queue", "amqp", "exchange"],
            
            # Load balancing and networking
            "nginx": ["reverse proxy", "load balancer", "upstream"],
            "haproxy": ["load balancing", "health check"],
            
            # Monitoring and observability
            "prometheus": ["metrics", "scraping", "alerting"],
            "grafana": ["dashboard", "visualization", "alerting"],
            
            # CI/CD and deployment
            "jenkins": ["pipeline", "build", "deployment"],
            "terraform": ["infrastructure as code", "provisioning"]
        }
    
    def classify_failure_pattern(self, content: str, title: str) -> Optional[str]:
        """Classify the failure pattern for pattern discovery"""
        combined_text = (title + " " + content).lower()
        
        # Score each pattern based on keyword matches
        pattern_scores = {}
        for pattern, keywords in self.failure_patterns.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                pattern_scores[pattern] = score
        
        # Return the highest scoring pattern
        if pattern_scores:
            return max(pattern_scores, key=pattern_scores.get)
        return None
    
    def extract_timeline_events(self, content: str) -> List[Dict[str, str]]:
        """Extract timeline events from postmortem content"""
        events = []
        
        # Look for timestamp patterns
        timestamp_patterns = [
            r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\s*[-:]\s*(.+?)(?:\n|$)',
            r'(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?)\s*[-:]\s*(.+?)(?:\n|$)',
            r'(T\+\d+(?:h|m|s))\s*[-:]\s*(.+?)(?:\n|$)',
            r'(\d+\s*minutes?\s*(?:in|later))\s*[-:]\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in timestamp_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                timestamp, event = match.groups()
                if len(event.strip()) > 10:  # Filter out very short events
                    events.append({
                        "timestamp": timestamp.strip(),
                        "event": event.strip()[:200]  # Limit length
                    })
        
        return events[:20]  # Limit to 20 events
    
    def extract_mitigation_actions(self, content: str) -> List[str]:
        """Extract mitigation actions taken"""
        actions = []
        content_lower = content.lower()
        
        # Look for action indicators
        action_patterns = [
            r'(?:fixed by|resolved by|mitigated by|action taken)[:\-]?\s*(.+?)(?:\n|\.)',
            r'(?:we|team|engineer)?\s*(?:restarted|scaled|deployed|rolled back|reverted|increased|decreased)\s+(.+?)(?:\n|\.)',
            r'(?:immediate action|short term|long term)[:\-]?\s*(.+?)(?:\n|\.)',
        ]
        
        for pattern in action_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                action = match.group(1).strip()
                if len(action) > 5 and len(action) < 150:
                    actions.append(action)
        
        return list(set(actions))[:10]  # Deduplicate and limit
    
    def calculate_quality_score(self, content: str, title: str, metadata: Dict[str, Any]) -> float:
        """Calculate quality score for pattern analysis relevance"""
        score = 0.0
        
        # Length indicators (longer content often better for analysis)
        content_length = len(content)
        if content_length > 2000:
            score += 0.3
        elif content_length > 1000:
            score += 0.2
        elif content_length > 500:
            score += 0.1
        
        # Pattern keywords present
        if metadata.get("failure_pattern"):
            score += 0.3
        
        # Infrastructure components mentioned
        infra_count = len(metadata.get("infrastructure_components", []))
        score += min(infra_count * 0.05, 0.2)
        
        # Timeline events present
        timeline_count = len(metadata.get("timeline_events", []))
        score += min(timeline_count * 0.02, 0.1)
        
        # Severity indicators
        if metadata.get("severity") == "high":
            score += 0.1
        
        # Quality keywords
        quality_keywords = [
            "root cause", "postmortem", "incident", "analysis", 
            "timeline", "mitigation", "action items", "lessons learned"
        ]
        quality_score = sum(1 for keyword in quality_keywords if keyword in content.lower())
        score += min(quality_score * 0.02, 0.1)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def extract_postmortem_metadata(self, content: str, source_url: str, title: str = "") -> Dict[str, Any]:
        """Enhanced metadata extraction for pattern discovery"""
        metadata = {
            "tags": [],
            "services_affected": [],
            "infrastructure_components": [],
            "timeline_events": [],
            "mitigation_actions": []
        }
        
        content_lower = content.lower()
        
        # Extract infrastructure components with context
        for component, related_terms in self.infra_keywords.items():
            if component in content_lower or any(term in content_lower for term in related_terms):
                metadata["infrastructure_components"].append(component)
        
        # Classify failure pattern
        metadata["failure_pattern"] = self.classify_failure_pattern(content, title)
        
        # Extract timeline events
        metadata["timeline_events"] = self.extract_timeline_events(content)
        
        # Extract mitigation actions
        metadata["mitigation_actions"] = self.extract_mitigation_actions(content)
        
        # Enhanced severity classification
        if any(word in content_lower for word in ["critical", "severe", "outage", "down", "disaster", "emergency"]):
            metadata["severity"] = "high"
        elif any(word in content_lower for word in ["degraded", "slow", "performance", "minor", "warning"]):
            metadata["severity"] = "medium"
        else:
            metadata["severity"] = "low"
        
        # Blast radius extraction
        if any(word in content_lower for word in ["all users", "entire system", "global", "complete"]):
            metadata["blast_radius"] = "global"
        elif any(word in content_lower for word in ["subset", "partial", "some users", "region"]):
            metadata["blast_radius"] = "partial"
        else:
            metadata["blast_radius"] = "localized"
        
        # Extract service types with more precision
        service_indicators = {
            "api": ["api", "rest", "graphql", "endpoint"],
            "web": ["website", "frontend", "ui", "web app"],
            "database": ["database", "db", "storage", "persistence"],
            "cache": ["cache", "redis", "memcache"],
            "queue": ["queue", "messaging", "async"],
            "auth": ["authentication", "authorization", "login", "oauth"],
            "monitoring": ["monitoring", "metrics", "alerts", "observability"]
        }
        
        for service_type, indicators in service_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                metadata["services_affected"].append(service_type)
        
        # Calculate quality score
        metadata["quality_score"] = self.calculate_quality_score(content, title, metadata)
        
        return metadata

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_github_repo_contents(self, client: httpx.AsyncClient, repo: str) -> List[Dict[str, Any]]:
        """Fetch repository contents from GitHub API"""
        url = f"https://api.github.com/repos/{repo}/contents"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to fetch {repo}: {e}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_file_content(self, client: httpx.AsyncClient, download_url: str) -> Optional[str]:
        """Fetch individual file content"""
        try:
            response = await client.get(download_url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Failed to fetch file content: {e}")
            return None
    
    def is_high_quality_postmortem(self, filename: str, content: str) -> bool:
        """Enhanced filtering for high-quality postmortems"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Must have postmortem-related filename
        filename_indicators = [
            "postmortem", "incident", "outage", "failure", "issue", 
            "post-mortem", "retrospective", "analysis", "investigation"
        ]
        
        if not any(indicator in filename_lower for indicator in filename_indicators):
            return False
        
        # ENHANCEMENT 2: Lower length threshold from 500 to 300
        if len(content) < 300:
            return False
        
        # Must mention infrastructure or technical components
        tech_indicators = [
            "server", "database", "service", "api", "system", "infrastructure",
            "kubernetes", "docker", "aws", "gcp", "azure", "redis", "nginx",
            "deployment", "monitoring", "load balancer", "cache"
        ]
        
        if not any(indicator in content_lower for indicator in tech_indicators):
            return False
        
        # Must have incident-related keywords
        incident_indicators = [
            "incident", "outage", "downtime", "failure", "error", "issue",
            "problem", "root cause", "timeline", "resolution", "mitigation"
        ]
        
        if not any(indicator in content_lower for indicator in incident_indicators):
            return False
        
        return True
    
    async def harvest_github_postmortems(self, max_files: int = 1000) -> List[PostmortemData]:
        """Enhanced harvesting from GitHub repositories"""
        postmortems = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for repo in tqdm(self.github_repos, desc="Scanning high-quality repos"):
                try:
                    contents = await self.fetch_github_repo_contents(client, repo)
                    
                    # Enhanced file filtering
                    candidate_files = []
                    for item in contents:
                        if item.get("type") == "file" and item.get("name", "").endswith((".md", ".txt", ".rst")):
                            candidate_files.append(item)
                    
                    # Also check subdirectories for postmortems
                    for item in contents:
                        if item.get("type") == "dir" and any(keyword in item.get("name", "").lower() 
                                                           for keyword in ["incident", "postmortem", "outage", "docs"]):
                            try:
                                subdir_contents = await self.fetch_github_repo_contents(client, f"{repo}/{item['name']}")
                                for subitem in subdir_contents:
                                    if subitem.get("type") == "file" and subitem.get("name", "").endswith((".md", ".txt", ".rst")):
                                        candidate_files.append(subitem)
                            except:
                                continue
                    
                    # Process files with enhanced filtering
                    processed_count = 0
                    for file_info in candidate_files:
                        if processed_count >= max_files // len(self.github_repos):
                            break
                            
                        content = await self.fetch_file_content(client, file_info["download_url"])
                        if content and self.is_high_quality_postmortem(file_info["name"], content):
                            title = file_info["name"]
                            metadata = self.extract_postmortem_metadata(content, file_info["html_url"], title)
                            
                            # ENHANCEMENT 2: Lower quality threshold to 0.25
                            if metadata["quality_score"] >= 0.25:
                                postmortem = PostmortemData(
                                    source=f"github:{repo}",
                                    url=file_info["html_url"],
                                    title=title,
                                    content=content,
                                    timestamp=datetime.now().isoformat(),
                                    tags=metadata["tags"],
                                    severity=metadata["severity"],
                                    services_affected=metadata["services_affected"],
                                    infrastructure_components=metadata["infrastructure_components"],
                                    failure_pattern=metadata["failure_pattern"],
                                    timeline_events=metadata["timeline_events"],
                                    blast_radius=metadata["blast_radius"],
                                    mitigation_actions=metadata["mitigation_actions"],
                                    quality_score=metadata["quality_score"]
                                )
                                postmortems.append(postmortem)
                                processed_count += 1
                            
                        # Rate limiting
                        await asyncio.sleep(0.3)
                        
                except Exception as e:
                    logger.error(f"Error processing repo {repo}: {e}")
                    continue
        
        return postmortems
    
    async def search_github_issues(self, query: str = "postmortem OR incident OR outage infrastructure", max_results: int = 200) -> List[PostmortemData]:
        """Enhanced GitHub issues search with better filtering"""
        postmortems = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Enhanced search queries for different types of infrastructure incidents
            search_queries = [
                "postmortem infrastructure in:title,body type:issue",
                "incident analysis infrastructure in:title,body type:issue", 
                "outage root cause in:title,body type:issue",
                "system failure kubernetes docker in:title,body type:issue",
                "production incident database in:title,body type:issue"
            ]
            
            for search_query in search_queries:
                try:
                    params = {
                        "q": search_query,
                        "sort": "updated",
                        "order": "desc",
                        "per_page": min(50, max_results // len(search_queries))
                    }
                    
                    response = await client.get("https://api.github.com/search/issues", headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    for issue in data.get("items", []):
                        if issue.get("body") and len(issue["body"]) > 300:  # ENHANCEMENT 2: Lower length threshold
                            title = issue["title"]
                            content = issue["body"]
                            
                            if self.is_high_quality_postmortem(title, content):
                                metadata = self.extract_postmortem_metadata(content, issue["html_url"], title)
                                
                                # Only include high-quality issues
                                if metadata["quality_score"] >= 0.4:  # Higher threshold for issues
                                    postmortem = PostmortemData(
                                        source="github:issues",
                                        url=issue["html_url"],
                                        title=title,
                                        content=content,
                                        timestamp=issue["updated_at"],
                                        tags=metadata["tags"],
                                        severity=metadata["severity"],
                                        services_affected=metadata["services_affected"],
                                        infrastructure_components=metadata["infrastructure_components"],
                                        failure_pattern=metadata["failure_pattern"],
                                        timeline_events=metadata["timeline_events"],
                                        blast_radius=metadata["blast_radius"],
                                        mitigation_actions=metadata["mitigation_actions"],
                                        quality_score=metadata["quality_score"]
                                    )
                                    postmortems.append(postmortem)
                    
                    await asyncio.sleep(1.0)  # Rate limiting between queries
                    
                except Exception as e:
                    logger.warning(f"Error searching GitHub issues with query '{search_query}': {e}")
                    continue
        
        return postmortems

    def save_postmortems(self, postmortems: List[PostmortemData], filename: str = None) -> str:
        """Save harvested postmortems to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"postmortems_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        data = [asdict(pm) for pm in postmortems]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Also save as parquet for faster processing
        if data:
            df = pd.DataFrame(data)
            parquet_path = output_path.with_suffix('.parquet')
            df.to_parquet(parquet_path, index=False, engine='fastparquet')
        
        logger.info(f"Saved {len(postmortems)} postmortems to {output_path}")
        return str(output_path)
    
    async def github_code_search_enhanced(self, max_results: int = 400) -> List[PostmortemData]:
        """🔍 Enhanced GitHub CodeSearch with proper pagination and deduplication"""
        postmortems = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Enhanced targeted search queries for high-quality infrastructure postmortems
            search_queries = [
                # Target specific high-value organizations and repositories
                "postmortem org:kubernetes language:Markdown",
                "postmortem org:prometheus language:Markdown", 
                "postmortem org:grafana language:Markdown",
                "incident org:kubernetes language:Markdown",
                "incident org:prometheus language:Markdown",
                "incident org:grafana language:Markdown",
                
                # Target specific infrastructure components with postmortems
                "postmortem docker language:Markdown size:>1000",
                "postmortem redis language:Markdown size:>1000", 
                "postmortem nginx language:Markdown size:>1000",
                "postmortem elasticsearch language:Markdown size:>1000",
                "postmortem mysql language:Markdown size:>1000",
                "postmortem postgres language:Markdown size:>1000",
                
                # Target outage and failure analysis
                "outage analysis language:Markdown size:>800",
                "incident report language:Markdown size:>800",
                "root cause language:Markdown size:>800",
                "system failure language:Markdown size:>800",
                "service disruption language:Markdown size:>800",
                
                # Target specific failure patterns
                "cascading failure language:Markdown",
                "split brain language:Markdown", 
                "network partition language:Markdown",
                "memory leak incident language:Markdown",
                "database corruption language:Markdown",
                
                # Target infrastructure companies and SRE content
                "postmortem org:hashicorp language:Markdown",
                "postmortem org:elastic language:Markdown",
                "postmortem org:spotify language:Markdown",
                "incident org:netflix language:Markdown",
                "SRE postmortem language:Markdown",
                
                # General high-quality postmortem patterns
                "filename:postmortem extension:md size:>1000",
                "filename:incident extension:md size:>1000",
                "path:docs postmortem language:Markdown",
                "path:incidents language:Markdown"
            ]
            
            for query in search_queries:
                try:
                    logger.info(f"🔍 Searching: {query}")
                    page = 1
                    query_results = 0
                    max_pages = 3  # Increased for better coverage
                    
                    while page <= max_pages and query_results < (max_results // len(search_queries)):
                        params = {
                            "q": query,
                            "sort": "indexed",
                            "order": "desc", 
                            "per_page": 30,  # Smaller pages for better success rate
                            "page": page
                        }
                        
                        logger.debug(f"Making request with params: {params}")
                        response = await client.get("https://api.github.com/search/code", headers=headers, params=params)
                        
                        logger.debug(f"Response status: {response.status_code}")
                        if response.status_code == 422:
                            logger.warning(f"Query syntax issue: {query} - {response.text}")
                            break
                        elif response.status_code == 403:
                            logger.warning(f"Rate limited, sleeping... - {response.text}")
                            await asyncio.sleep(10)
                            continue
                        
                        response.raise_for_status()
                        data = response.json()
                        logger.debug(f"Found {len(data.get('items', []))} items for query: {query}")
                        
                        items = data.get("items", [])
                        if not items:
                            break
                        
                        for item in items:
                            try:
                                # Construct proper raw content URL from repository info
                                html_url = item.get("html_url", "")
                                if not html_url:
                                    continue
                                
                                # Convert GitHub HTML URL to raw content URL
                                # e.g., https://github.com/user/repo/blob/main/file.md -> https://raw.githubusercontent.com/user/repo/main/file.md
                                if "github.com" in html_url and "/blob/" in html_url:
                                    raw_url = html_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                                else:
                                    # Fallback: try to use the download_url from GitHub API
                                    raw_url = item.get("download_url")
                                    if not raw_url:
                                        continue
                                
                                # Check for duplicates
                                if self.seen_index.is_duplicate(raw_url, ""):
                                    continue
                                
                                # Fetch file content
                                content = await self.fetch_file_content(client, raw_url)
                                if not content:
                                    continue
                                
                                # Check for content duplicates
                                if self.seen_index.is_duplicate(raw_url, content):
                                    continue
                                
                                filename = item.get("name", "")
                                if content and self.is_high_quality_postmortem(filename, content):
                                    metadata = self.extract_postmortem_metadata(content, item.get("html_url", raw_url), filename)
                                    
                                    if metadata["quality_score"] >= 0.35:
                                        # Add to seen index
                                        self.seen_index.add(raw_url, content)
                                        
                                        postmortem = PostmortemData(
                                            source="github:codesearch",
                                            url=item.get("html_url", raw_url),
                                            title=filename,
                                            content=content,
                                            timestamp=datetime.now().isoformat(),
                                            tags=metadata["tags"],
                                            severity=metadata["severity"],
                                            services_affected=metadata["services_affected"],
                                            infrastructure_components=metadata["infrastructure_components"],
                                            failure_pattern=metadata["failure_pattern"],
                                            timeline_events=metadata["timeline_events"],
                                            blast_radius=metadata["blast_radius"],
                                            mitigation_actions=metadata["mitigation_actions"],
                                            quality_score=metadata["quality_score"]
                                        )
                                        postmortems.append(postmortem)
                                        query_results += 1
                                        
                                        if len(postmortems) >= max_results:
                                            logger.info(f"✅ Reached max results limit: {max_results}")
                                            return postmortems
                                
                                await asyncio.sleep(0.2)  # Rate limiting
                                
                            except Exception as e:
                                logger.warning(f"Failed to process code search result: {e}")
                                continue
                        
                        page += 1
                        await asyncio.sleep(1.0)  # Rate limiting between pages
                        
                        if not data.get("incomplete_results", False):
                            break
                
                except Exception as e:
                    logger.warning(f"Code search failed for query '{query}': {e}")
                    continue
        
        logger.info(f"✅ GitHub CodeSearch found {len(postmortems)} postmortems")
        logger.info(f"📊 Deduplication stats: {self.seen_index.stats()}")
        return postmortems

    async def harvest_all(self, max_postmortems: int = 1000) -> List[PostmortemData]:
        """Enhanced harvesting with pattern discovery focus"""
        logger.info("🔍 Starting enhanced postmortem harvest for pattern discovery")
        
        all_postmortems = []
        
        # ENHANCEMENT 1: GitHub CodeSearch for wider coverage (400 docs)
        logger.info("🔍 Using GitHub CodeSearch for wider coverage...")
        codesearch_postmortems = await self.github_code_search_enhanced(max_results=400)
        all_postmortems.extend(codesearch_postmortems)
        
        # Harvest from curated GitHub repositories
        logger.info("Harvesting from curated high-quality repositories...")
        repo_postmortems = await self.harvest_github_postmortems(max_postmortems // 3)
        all_postmortems.extend(repo_postmortems)
        
        # Search GitHub issues for additional incidents
        logger.info("Searching GitHub issues for incident reports...")
        issue_postmortems = await self.search_github_issues(max_results=max_postmortems // 3)
        all_postmortems.extend(issue_postmortems)
        
        # Enhanced deduplication based on content similarity
        unique_postmortems = []
        seen_urls = set()
        content_hashes = set()
        
        for pm in sorted(all_postmortems, key=lambda x: x.quality_score, reverse=True):
            # Skip if we've seen this URL
            if pm.url in seen_urls:
                continue
                
            # Simple content deduplication
            content_hash = hash(pm.content[:500])  # Use first 500 chars for similarity
            if content_hash in content_hashes:
                continue
                
            unique_postmortems.append(pm)
            seen_urls.add(pm.url)
            content_hashes.add(content_hash)
        
        # Sort by quality score for pattern discovery
        unique_postmortems.sort(key=lambda x: x.quality_score, reverse=True)
        
        logger.info(f"✅ Harvested {len(unique_postmortems)} unique high-quality postmortems")
        
        # Log pattern distribution
        if unique_postmortems:
            patterns = [pm.failure_pattern for pm in unique_postmortems if pm.failure_pattern]
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            logger.info(f"📊 Failure patterns discovered: {pattern_counts}")
            logger.info(f"📊 Average quality score: {sum(pm.quality_score for pm in unique_postmortems) / len(unique_postmortems):.2f}")
        
        return unique_postmortems

    async def codesearch_only(self, max_results: int = 300) -> List[PostmortemData]:
        """Run only GitHub Code Search for targeted harvesting"""
        logger.info(f"🔍 Starting GitHub Code Search only (target: {max_results} docs)")
        
        postmortems = await self.github_code_search_enhanced(max_results=max_results)
        
        logger.info(f"✅ GitHub Code Search completed: {len(postmortems)} postmortems found")
        
        # Log quality statistics
        if postmortems:
            patterns = [pm.failure_pattern for pm in postmortems if pm.failure_pattern]
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            logger.info(f"📊 Failure patterns discovered: {pattern_counts}")
            logger.info(f"📊 Average quality score: {sum(pm.quality_score for pm in postmortems) / len(postmortems):.2f}")
        
        return postmortems


def run_cli_async(codesearch: bool, limit: int, output_dir: str):
    """Async wrapper for CLI"""
    return asyncio.run(cli_main_async(codesearch, limit, output_dir))


async def cli_main_async(codesearch: bool, limit: int, output_dir: str):
    """CLI main function with GitHub Code Search support"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.error("GITHUB_TOKEN not found in environment variables")
        return
    
    # Initialize harvester
    harvester = PostmortemHarvester(github_token, output_dir)
    start_time = time.time()
    
    try:
        if codesearch:
            # Run GitHub Code Search only
            logger.info(f"🔍 Running GitHub Code Search with limit: {limit}")
            postmortems = await harvester.codesearch_only(max_results=limit)
        else:
            # Run full harvest
            logger.info(f"🔍 Running full harvest with limit: {limit}")
            postmortems = await harvester.harvest_all(limit)
        
        # Save results
        if postmortems:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "codesearch" if codesearch else "full"
            filename = f"postmortems_{suffix}_{timestamp}.json"
            output_file = harvester.save_postmortems(postmortems, filename)
            
            # Generate summary statistics
            df = pd.DataFrame([asdict(pm) for pm in postmortems])
            
            logger.info("📊 Enhanced Harvest Summary:")
            logger.info(f"   Total postmortems: {len(postmortems)}")
            logger.info(f"   Sources: {df['source'].nunique()}")
            logger.info(f"   High severity: {len(df[df['severity'] == 'high'])}")
            logger.info(f"   With failure patterns: {len(df[df['failure_pattern'].notna()])}")
            logger.info(f"   Average quality score: {df['quality_score'].mean():.2f}")
            logger.info(f"   Top failure patterns: {df['failure_pattern'].value_counts().head(3).to_dict()}")
            logger.info(f"   Output file: {output_file}")
        else:
            logger.warning("❌ No postmortems found")
        
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️  Harvest completed in {elapsed_time:.1f}s")
        
    except Exception as e:
        logger.error(f"❌ Harvest failed: {e}")
        raise


async def main():
    """Regular main function for backwards compatibility"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.error("GITHUB_TOKEN not found in environment variables")
        return
    
    max_postmortems = int(os.getenv("MAX_POSTMORTEMS", 1000))
    output_dir = os.getenv("DATA_DIR", "data/raw") + "/postmortems"
    
    # Initialize harvester
    harvester = PostmortemHarvester(github_token, output_dir)
    start_time = time.time()
    
    try:
        # Harvest postmortems
        postmortems = await harvester.harvest_all(max_postmortems)
        
        # Save results
        output_file = harvester.save_postmortems(postmortems)
        
        # Generate summary statistics
        if postmortems:
            df = pd.DataFrame([asdict(pm) for pm in postmortems])
            
            logger.info("📊 Enhanced Harvest Summary:")
            logger.info(f"   Total postmortems: {len(postmortems)}")
            logger.info(f"   Sources: {df['source'].nunique()}")
            logger.info(f"   High severity: {len(df[df['severity'] == 'high'])}")
            logger.info(f"   With failure patterns: {len(df[df['failure_pattern'].notna()])}")
            logger.info(f"   Average quality score: {df['quality_score'].mean():.2f}")
            logger.info(f"   Top failure patterns: {df['failure_pattern'].value_counts().head(3).to_dict()}")
            logger.info(f"   Output file: {output_file}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️  Enhanced harvest completed in {elapsed_time:.1f}s")
        
    except Exception as e:
        logger.error(f"❌ Harvest failed: {e}")
        raise


@click.command()
@click.option('--codesearch', is_flag=True, help='Use GitHub Code Search only')
@click.option('--limit', default=300, type=int, help='Maximum number of documents to harvest')
@click.option('--output-dir', default="data/raw/postmortems", help='Output directory')
def cli_command(codesearch: bool, limit: int, output_dir: str):
    """CLI command entry point"""
    run_cli_async(codesearch, limit, output_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and ('--codesearch' in sys.argv or '--limit' in sys.argv or '--help' in sys.argv):
        # Use CLI interface
        cli_command()
    else:
        # Use regular async main
        asyncio.run(main())
