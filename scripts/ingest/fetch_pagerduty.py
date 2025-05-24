#!/usr/bin/env python3
"""
ENHANCEMENT 4: Import PagerDuty Postmortem Library
Part of Protean Pattern Discovery Engine - 10× Corpus Enhancement
"""

import os
import json
import time
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from loguru import logger


@dataclass
class PagerDutyPost:
    """Structure for PagerDuty postmortem data"""
    source: str
    url: str
    title: str
    content: str
    timestamp: str
    tags: List[str]
    severity: Optional[str] = None
    services_affected: List[str] = None
    infrastructure_components: List[str] = None
    failure_pattern: Optional[str] = None
    timeline_events: List[Dict[str, str]] = None
    blast_radius: Optional[str] = None
    mitigation_actions: List[str] = None
    quality_score: float = 0.0


class PagerDutyHarvester:
    """Harvest postmortem content from PagerDuty public learning center"""
    
    def __init__(self, output_dir: str = "data/raw/pagerduty"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # PagerDuty learning center and resources URLs
        self.pagerduty_sources = [
            # Learning center JSON APIs and endpoints
            "https://www.pagerduty.com/api/learning-center/postmortems",
            "https://www.pagerduty.com/resources/learn/post-incident-review/",
            "https://www.pagerduty.com/resources/learn/what-is-a-postmortem/",
            
            # Known PagerDuty postmortem repositories and case studies
            "https://postmortems.pagerduty.com/",
            "https://www.pagerduty.com/case-studies/",
            "https://community.pagerduty.com/c/postmortems/",
            
            # PagerDuty incident response guides
            "https://response.pagerduty.com/",
            "https://www.pagerduty.com/resources/incident-response/",
        ]
        
        # Alternative approach: scrape known PagerDuty content
        self.content_patterns = [
            "incident-response",
            "postmortem",
            "post-incident",
            "case-study",
            "outage-report"
        ]
        
        # Infrastructure components for classification
        self.infra_keywords = {
            "kubernetes": ["k8s", "pod", "deployment", "service mesh"],
            "docker": ["container", "containerization", "registry"],
            "aws": ["ec2", "s3", "rds", "lambda", "cloudformation"],
            "gcp": ["gke", "compute engine", "cloud sql"],
            "azure": ["vm", "blob storage", "cosmos db"],
            "postgresql": ["postgres", "pg", "database"],
            "redis": ["cache", "session store"],
            "nginx": ["reverse proxy", "load balancer"],
            "kafka": ["streaming", "topic", "partition"],
            "elasticsearch": ["search", "indexing", "cluster"],
            "prometheus": ["metrics", "scraping", "alerting"],
            "grafana": ["dashboard", "visualization"],
            "pagerduty": ["incident", "alert", "escalation", "on-call"]
        }
        
        # Failure pattern classification
        self.failure_patterns = {
            "cascading_failure": ["cascade", "cascading", "domino effect", "chain reaction"],
            "resource_exhaustion": ["memory leak", "cpu spike", "out of memory", "capacity"],
            "network_partition": ["network partition", "split brain", "connectivity"],
            "configuration_drift": ["configuration", "config drift", "misconfiguration"],
            "dependency_failure": ["upstream", "downstream", "third party", "external service"],
            "data_corruption": ["data corruption", "data loss", "inconsistent state"],
            "scaling_failure": ["auto scaling", "horizontal scaling", "load balancer"],
            "monitoring_blind_spot": ["monitoring", "alerting", "no visibility", "silent failure"]
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_pagerduty_content(self, client: httpx.AsyncClient, url: str) -> Optional[Dict[str, Any]]:
        """Fetch content from PagerDuty URLs"""
        try:
            response = await client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                # Handle JSON responses
                return {"type": "json", "data": response.json(), "url": url}
            else:
                # Handle HTML content
                html_content = response.text
                
                # Extract text content (basic HTML parsing)
                import re
                clean_content = re.sub(r'<[^>]+>', ' ', html_content)
                clean_content = re.sub(r'\s+', ' ', clean_content)
                
                # Extract title from HTML
                title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
                title = title_match.group(1).strip() if title_match else "PagerDuty Content"
                
                return {
                    "type": "html",
                    "data": {
                        "title": title,
                        "content": clean_content,
                        "url": url
                    },
                    "url": url
                }
                
        except Exception as e:
            logger.warning(f"Failed to fetch PagerDuty content from {url}: {e}")
            return None
    
    async def search_pagerduty_github(self, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        """Search for PagerDuty postmortem content on GitHub"""
        github_results = []
        
        # Search for PagerDuty-related postmortems on GitHub
        search_queries = [
            "pagerduty postmortem in:readme,description",
            "pagerduty incident response in:readme,description",  
            "pagerduty case study in:readme,description",
            "org:pagerduty postmortem",
            "org:pagerduty incident"
        ]
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ProteanHarvester/1.0"
        }
        
        for query in search_queries:
            try:
                params = {
                    "q": query,
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 10
                }
                
                response = await client.get("https://api.github.com/search/repositories", headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    github_results.extend(data.get("items", []))
                
                await asyncio.sleep(1.0)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"GitHub search failed for query '{query}': {e}")
                continue
        
        return github_results
    
    def extract_metadata(self, title: str, content: str, url: str) -> Dict[str, Any]:
        """Extract metadata from PagerDuty content"""
        metadata = {
            "tags": ["pagerduty"],
            "services_affected": [],
            "infrastructure_components": [],
            "timeline_events": [],
            "mitigation_actions": []
        }
        
        content_lower = content.lower()
        title_lower = title.lower()
        combined = title_lower + " " + content_lower
        
        # Extract infrastructure components
        for component, keywords in self.infra_keywords.items():
            if component in combined or any(keyword in combined for keyword in keywords):
                metadata["infrastructure_components"].append(component)
        
        # Classify failure pattern
        for pattern, keywords in self.failure_patterns.items():
            if any(keyword in combined for keyword in keywords):
                metadata["failure_pattern"] = pattern
                break
        
        # Extract severity
        if any(word in combined for word in ["critical", "severe", "major", "outage", "emergency"]):
            metadata["severity"] = "high"
        elif any(word in combined for word in ["minor", "degraded", "partial", "warning"]):
            metadata["severity"] = "medium"
        else:
            metadata["severity"] = "low"
        
        # Extract affected services
        service_indicators = {
            "api": ["api", "rest", "endpoint", "service"],
            "web": ["website", "frontend", "web", "ui"],
            "database": ["database", "db", "storage", "data"],
            "monitoring": ["monitoring", "metrics", "alerts", "pagerduty"],
            "auth": ["authentication", "authorization", "login"],
            "queue": ["queue", "messaging", "async"]
        }
        
        for service, indicators in service_indicators.items():
            if any(indicator in combined for indicator in indicators):
                metadata["services_affected"].append(service)
        
        # Blast radius
        if any(word in combined for word in ["all users", "global", "entire", "complete", "organization"]):
            metadata["blast_radius"] = "global"
        elif any(word in combined for word in ["partial", "some", "subset", "region"]):
            metadata["blast_radius"] = "partial"
        else:
            metadata["blast_radius"] = "localized"
        
        # Extract timeline events (PagerDuty specific patterns)
        timeline_patterns = [
            r'(\d{1,2}:\d{2}(?:\s*[AP]M)?)\s*[-:]\s*(.+?)(?:\n|$)',
            r'(Step \d+)[:\-]\s*(.+?)(?:\n|$)',
            r'(\d+\s*min(?:ute)?s?)[:\-]\s*(.+?)(?:\n|$)'
        ]
        
        events = []
        for pattern in timeline_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                timestamp, event = match.groups()
                if len(event.strip()) > 10:
                    events.append({
                        "timestamp": timestamp.strip(),
                        "event": event.strip()[:200]
                    })
        
        metadata["timeline_events"] = events[:15]
        
        # Extract mitigation actions
        action_patterns = [
            r'(?:action|step|solution)[:\-]\s*(.+?)(?:\n|\.)',
            r'(?:implemented|deployed|fixed)[:\-]?\s*(.+?)(?:\n|\.)',
            r'(?:to resolve|to fix|remediation)[:\-]?\s*(.+?)(?:\n|\.)'
        ]
        
        actions = []
        for pattern in action_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                action = match.group(1).strip()
                if len(action) > 5 and len(action) < 150:
                    actions.append(action)
        
        metadata["mitigation_actions"] = list(set(actions))[:10]
        
        # Calculate quality score
        quality_score = 0.0
        
        # Content length bonus
        if len(content) > 2000:
            quality_score += 0.4
        elif len(content) > 1000:
            quality_score += 0.3
        elif len(content) > 500:
            quality_score += 0.2
        
        # Infrastructure components bonus
        quality_score += min(len(metadata["infrastructure_components"]) * 0.1, 0.3)
        
        # Pattern classification bonus
        if metadata.get("failure_pattern"):
            quality_score += 0.2
        
        # PagerDuty-specific keywords bonus
        pd_keywords = ["incident", "postmortem", "response", "escalation", "on-call", "alert"]
        pd_count = sum(1 for keyword in pd_keywords if keyword in combined)
        quality_score += min(pd_count * 0.03, 0.15)
        
        # Timeline events bonus
        quality_score += min(len(metadata["timeline_events"]) * 0.02, 0.1)
        
        metadata["quality_score"] = min(quality_score, 1.0)
        
        return metadata
    
    def is_postmortem_content(self, title: str, content: str) -> bool:
        """Check if content appears to be a postmortem"""
        combined_text = (title + " " + content).lower()
        
        # Must contain postmortem-related keywords
        postmortem_keywords = [
            "postmortem", "post-mortem", "incident", "outage", "failure",
            "root cause", "analysis", "investigation", "case study",
            "lessons learned", "post-incident", "retrospective"
        ]
        
        has_postmortem_keywords = any(keyword in combined_text for keyword in postmortem_keywords)
        
        # Must have sufficient content  
        has_sufficient_content = len(content) > 300  # Match lowered threshold
        
        # Must mention technical or infrastructure terms
        tech_terms = ["service", "system", "infrastructure", "database", "server", "api", 
                     "application", "network", "deployment", "monitoring", "alert"]
        has_tech_content = any(term in combined_text for term in tech_terms)
        
        return has_postmortem_keywords and has_sufficient_content and has_tech_content
    
    async def harvest_pagerduty_content(self, max_posts: int = 30) -> List[PagerDutyPost]:
        """Harvest postmortem content from PagerDuty sources"""
        posts = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # First, try to fetch from known PagerDuty URLs
            for url in tqdm(self.pagerduty_sources, desc="Fetching PagerDuty content"):
                try:
                    content_data = await self.fetch_pagerduty_content(client, url)
                    
                    if content_data:
                        if content_data["type"] == "json":
                            # Handle JSON response
                            json_data = content_data["data"]
                            
                            # Extract postmortem entries from JSON
                            if isinstance(json_data, dict):
                                if "postmortems" in json_data:
                                    for item in json_data["postmortems"]:
                                        title = item.get("title", "PagerDuty Postmortem")
                                        content = item.get("content", "") or item.get("description", "")
                                        
                                        if self.is_postmortem_content(title, content):
                                            metadata = self.extract_metadata(title, content, url)
                                            
                                            if metadata["quality_score"] >= 0.25:
                                                post = PagerDutyPost(
                                                    source="pagerduty:api",
                                                    url=url,
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
                                                posts.append(post)
                        
                        elif content_data["type"] == "html":
                            # Handle HTML content
                            html_data = content_data["data"]
                            title = html_data["title"]
                            content = html_data["content"]
                            
                            if self.is_postmortem_content(title, content):
                                metadata = self.extract_metadata(title, content, url)
                                
                                if metadata["quality_score"] >= 0.25:
                                    post = PagerDutyPost(
                                        source="pagerduty:web",
                                        url=url,
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
                                    posts.append(post)
                    
                    await asyncio.sleep(1.0)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error processing PagerDuty URL {url}: {e}")
                    continue
            
            # Also search GitHub for PagerDuty-related content
            try:
                github_results = await self.search_pagerduty_github(client)
                
                for repo in github_results[:10]:  # Limit to top 10 repos
                    try:
                        # Fetch README and description
                        readme_url = f"https://raw.githubusercontent.com/{repo['full_name']}/main/README.md"
                        readme_response = await client.get(readme_url)
                        
                        if readme_response.status_code == 200:
                            readme_content = readme_response.text
                            title = repo.get("name", "PagerDuty GitHub Content")
                            description = repo.get("description", "")
                            combined_content = description + "\n\n" + readme_content
                            
                            if self.is_postmortem_content(title, combined_content):
                                metadata = self.extract_metadata(title, combined_content, repo["html_url"])
                                
                                if metadata["quality_score"] >= 0.25:
                                    post = PagerDutyPost(
                                        source="pagerduty:github",
                                        url=repo["html_url"],
                                        title=title,
                                        content=combined_content,
                                        timestamp=repo.get("updated_at", datetime.now().isoformat()),
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
                                    posts.append(post)
                        
                        await asyncio.sleep(1.0)
                        
                    except Exception as e:
                        logger.warning(f"Error processing GitHub repo {repo.get('full_name', 'unknown')}: {e}")
                        continue
            
            except Exception as e:
                logger.warning(f"GitHub search for PagerDuty content failed: {e}")
        
        # Sort by quality score and limit
        posts.sort(key=lambda x: x.quality_score, reverse=True)
        
        logger.info(f"✅ Harvested {len(posts)} PagerDuty postmortems")
        return posts[:max_posts]
    
    def save_posts(self, posts: List[PagerDutyPost], filename: str = None) -> str:
        """Save harvested posts to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pagerduty_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        data = [asdict(post) for post in posts]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Also save as parquet
        if data:
            df = pd.DataFrame(data)
            parquet_path = output_path.with_suffix('.parquet')
            df.to_parquet(parquet_path, index=False, engine='fastparquet')
        
        logger.info(f"Saved {len(posts)} PagerDuty posts to {output_path}")
        return str(output_path)


async def main():
    """Main harvesting function for PagerDuty"""
    from dotenv import load_dotenv
    load_dotenv()
    
    output_dir = os.getenv("DATA_DIR", "data/raw") + "/pagerduty"
    max_posts = 30  # Target: ~30 docs as specified
    
    harvester = PagerDutyHarvester(output_dir)
    
    try:
        posts = await harvester.harvest_pagerduty_content(max_posts)
        output_file = harvester.save_posts(posts)
        
        logger.info(f"✅ PagerDuty harvest complete: {len(posts)} posts saved to {output_file}")
        
    except Exception as e:
        logger.error(f"❌ PagerDuty harvest failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 