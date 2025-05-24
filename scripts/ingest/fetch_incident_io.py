#!/usr/bin/env python3
"""
ENHANCEMENT 3: Scrape incident.io Public RSS Feed for Postmortems
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
from xml.etree import ElementTree as ET

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from loguru import logger


@dataclass
class IncidentIOPost:
    """Structure for incident.io postmortem data"""
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


class IncidentIOHarvester:
    """Harvest postmortem content from incident.io public feeds"""
    
    def __init__(self, output_dir: str = "data/raw/incident_io"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # incident.io RSS feed URLs
        self.feed_urls = [
            "https://status.incident.io/feed",  # Main status feed
            "https://status.incident.io/incidents.rss",  # Alternative incident feed
        ]
        
        # Keywords to identify postmortem content
        self.postmortem_keywords = [
            "postmortem", "post-mortem", "incident report", "root cause",
            "analysis", "investigation", "retrospective", "lessons learned",
            "incident summary", "outage report"
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
            "grafana": ["dashboard", "visualization"]
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
    async def fetch_rss_feed(self, client: httpx.AsyncClient, feed_url: str) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed"""
        try:
            response = await client.get(feed_url)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.text)
            
            items = []
            
            # Handle different RSS formats
            if root.tag == "rss":
                # Standard RSS format
                for item in root.findall(".//item"):
                    title = item.find("title")
                    link = item.find("link") 
                    description = item.find("description")
                    pub_date = item.find("pubDate")
                    
                    items.append({
                        "title": title.text if title is not None else "",
                        "link": link.text if link is not None else "",
                        "description": description.text if description is not None else "",
                        "pub_date": pub_date.text if pub_date is not None else ""
                    })
            
            elif root.tag.endswith("feed"):
                # Atom feed format
                for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                    title = entry.find("{http://www.w3.org/2005/Atom}title")
                    link = entry.find("{http://www.w3.org/2005/Atom}link")
                    content = entry.find("{http://www.w3.org/2005/Atom}content")
                    updated = entry.find("{http://www.w3.org/2005/Atom}updated")
                    
                    items.append({
                        "title": title.text if title is not None else "",
                        "link": link.get("href") if link is not None else "",
                        "description": content.text if content is not None else "",
                        "pub_date": updated.text if updated is not None else ""
                    })
            
            logger.info(f"Fetched {len(items)} items from {feed_url}")
            return items
            
        except Exception as e:
            logger.warning(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_incident_details(self, client: httpx.AsyncClient, incident_url: str) -> Optional[str]:
        """Fetch full incident details from URL"""
        try:
            response = await client.get(incident_url)
            response.raise_for_status()
            
            # Extract text content (basic HTML parsing)
            content = response.text
            
            # Remove HTML tags for better text extraction
            import re
            clean_content = re.sub(r'<[^>]+>', ' ', content)
            clean_content = re.sub(r'\s+', ' ', clean_content)
            
            return clean_content
            
        except Exception as e:
            logger.warning(f"Failed to fetch incident details from {incident_url}: {e}")
            return None
    
    def is_postmortem_content(self, title: str, description: str) -> bool:
        """Check if content appears to be a postmortem"""
        combined_text = (title + " " + description).lower()
        
        # Must contain postmortem keywords
        has_postmortem_keywords = any(keyword in combined_text for keyword in self.postmortem_keywords)
        
        # Must have sufficient content
        has_sufficient_content = len(description) > 100
        
        # Must mention infrastructure or technical terms
        tech_terms = ["service", "system", "infrastructure", "database", "server", "api", 
                     "application", "network", "deployment", "monitoring"]
        has_tech_content = any(term in combined_text for term in tech_terms)
        
        return has_postmortem_keywords and has_sufficient_content and has_tech_content
    
    def extract_metadata(self, title: str, content: str, url: str) -> Dict[str, Any]:
        """Extract metadata from incident content"""
        metadata = {
            "tags": [],
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
        if any(word in combined for word in ["critical", "severe", "major", "outage"]):
            metadata["severity"] = "high"
        elif any(word in combined for word in ["minor", "degraded", "partial"]):
            metadata["severity"] = "medium"
        else:
            metadata["severity"] = "low"
        
        # Extract affected services
        service_indicators = {
            "api": ["api", "rest", "endpoint"],
            "web": ["website", "frontend", "web"],
            "database": ["database", "db", "storage"],
            "cache": ["cache", "redis"],
            "monitoring": ["monitoring", "metrics", "alerts"]
        }
        
        for service, indicators in service_indicators.items():
            if any(indicator in combined for indicator in indicators):
                metadata["services_affected"].append(service)
        
        # Blast radius
        if any(word in combined for word in ["all", "global", "entire", "complete"]):
            metadata["blast_radius"] = "global"
        elif any(word in combined for word in ["partial", "some", "subset"]):
            metadata["blast_radius"] = "partial"
        else:
            metadata["blast_radius"] = "localized"
        
        # Calculate quality score
        quality_score = 0.0
        
        # Content length bonus
        if len(content) > 1000:
            quality_score += 0.3
        elif len(content) > 500:
            quality_score += 0.2
        
        # Infrastructure components bonus
        quality_score += min(len(metadata["infrastructure_components"]) * 0.1, 0.3)
        
        # Pattern classification bonus
        if metadata.get("failure_pattern"):
            quality_score += 0.2
        
        # Postmortem keywords bonus
        postmortem_count = sum(1 for keyword in self.postmortem_keywords if keyword in combined)
        quality_score += min(postmortem_count * 0.05, 0.2)
        
        metadata["quality_score"] = min(quality_score, 1.0)
        
        return metadata
    
    async def harvest_incident_io_feeds(self, max_posts: int = 100) -> List[IncidentIOPost]:
        """Harvest postmortem content from incident.io feeds"""
        posts = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for feed_url in self.feed_urls:
                try:
                    feed_items = await self.fetch_rss_feed(client, feed_url)
                    
                    for item in tqdm(feed_items, desc=f"Processing {feed_url}"):
                        title = item.get("title", "")
                        description = item.get("description", "")
                        link = item.get("link", "")
                        pub_date = item.get("pub_date", "")
                        
                        # Filter for postmortem content using keyword matching
                        if "postmortem" in title.lower() or "postmortem" in description.lower():
                            # Fetch full incident details if available
                            full_content = await self.fetch_incident_details(client, link)
                            if full_content:
                                content = full_content
                            else:
                                content = description
                            
                            # Only include if content is substantial
                            if len(content) > 300:  # Match the lowered threshold
                                metadata = self.extract_metadata(title, content, link)
                                
                                # Quality threshold
                                if metadata["quality_score"] >= 0.25:  # Match lowered threshold
                                    post = IncidentIOPost(
                                        source="incident.io",
                                        url=link,
                                        title=title,
                                        content=content,
                                        timestamp=pub_date or datetime.now().isoformat(),
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
                        
                        # Rate limiting
                        await asyncio.sleep(0.5)
                        
                        if len(posts) >= max_posts:
                            break
                
                except Exception as e:
                    logger.error(f"Error processing feed {feed_url}: {e}")
                    continue
        
        logger.info(f"✅ Harvested {len(posts)} postmortems from incident.io")
        return posts
    
    def save_posts(self, posts: List[IncidentIOPost], filename: str = None) -> str:
        """Save harvested posts to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"incident_io_{timestamp}.json"
        
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
        
        logger.info(f"Saved {len(posts)} incident.io posts to {output_path}")
        return str(output_path)


async def main():
    """Main harvesting function for incident.io"""
    from dotenv import load_dotenv
    load_dotenv()
    
    output_dir = os.getenv("DATA_DIR", "data/raw") + "/incident_io"
    max_posts = 60  # Target: 40-60 docs as specified
    
    harvester = IncidentIOHarvester(output_dir)
    
    try:
        posts = await harvester.harvest_incident_io_feeds(max_posts)
        output_file = harvester.save_posts(posts)
        
        logger.info(f"✅ incident.io harvest complete: {len(posts)} posts saved to {output_file}")
        
    except Exception as e:
        logger.error(f"❌ incident.io harvest failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 