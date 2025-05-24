#!/usr/bin/env python3
"""
✨ Robust RSS Status Feed Harvester
Fetch postmortems from incident.io + PagerDuty RSS feeds
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
from urllib.parse import urljoin, urlparse

import httpx
import feedparser
import html2text
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from loguru import logger

# Import existing postmortem harvester for quality logic
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.ingest.fetch_postmortems import PostmortemHarvester, PostmortemData


@dataclass
class RSSPostmortem:
    """Structure for RSS-sourced postmortem data"""
    source: str
    url: str
    title: str
    content: str
    published: str
    summary: str
    tags: List[str] = None
    raw_html: str = ""


class StatusRSSHarvester:
    """Harvest postmortem content from status page RSS feeds"""
    
    def __init__(self, output_dir: str = "data/raw/postmortems"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # RSS feed URLs - verified working endpoints
        self.rss_feeds = {
            "incident.io": [
                "https://status.incident.io/feed",
                "https://status.incident.io/api/v1/incidents.rss",
                "https://incident.io/blog/rss.xml",
            ],
            "pagerduty": [
                "https://www.pagerduty.com/blog/category/post-mortem/feed/",
                "https://www.pagerduty.com/blog/feed/",
                "https://status.pagerduty.com/api/v2/incidents.rss",
            ],
            "statuspage": [
                "https://metastatuspage.com/api/v2/incidents.rss",
            ]
        }
        
        # Initialize HTML to markdown converter
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = True
        self.h2t.ignore_emphasis = False
        self.h2t.body_width = 0  # No line wrapping
        self.h2t.unicode_snob = True
        
        # Initialize postmortem harvester for quality assessment
        self.pm_harvester = PostmortemHarvester("dummy_token", str(self.output_dir))
        
        # Postmortem keywords for filtering
        self.postmortem_keywords = [
            "postmortem", "post-mortem", "incident", "outage", "downtime",
            "root cause", "analysis", "investigation", "failure", "service disruption",
            "retrospective", "lessons learned", "incident report", "post incident",
            "service degradation", "system failure", "infrastructure issue"
        ]
        
        # Quality thresholds
        self.min_content_length = 800
        self.min_quality_score = 0.3
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_rss_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed using feedparser"""
        try:
            logger.debug(f"Fetching RSS feed: {feed_url}")
            
            # Use httpx to fetch with proper headers
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Accept": "application/rss+xml, application/xml, text/xml, */*"
                }
                response = await client.get(feed_url, headers=headers)
                response.raise_for_status()
                
                # Parse RSS with feedparser
                feed = feedparser.parse(response.content)
                
                if feed.bozo:
                    logger.warning(f"RSS feed has parsing issues: {feed_url}")
                
                items = []
                for entry in feed.entries:
                    item = {
                        "title": getattr(entry, 'title', ''),
                        "link": getattr(entry, 'link', ''),
                        "summary": getattr(entry, 'summary', ''),
                        "published": getattr(entry, 'published', ''),
                        "content": self._extract_content_from_entry(entry),
                        "tags": self._extract_tags_from_entry(entry)
                    }
                    items.append(item)
                
                logger.info(f"✅ Fetched {len(items)} items from {feed_url}")
                return items
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch RSS feed {feed_url}: {e}")
            return []
    
    def _extract_content_from_entry(self, entry) -> str:
        """Extract content from RSS entry, trying multiple fields"""
        # Try different content fields
        content_fields = ['content', 'summary', 'description']
        
        for field in content_fields:
            if hasattr(entry, field):
                content_data = getattr(entry, field)
                if isinstance(content_data, list) and content_data:
                    return content_data[0].get('value', '')
                elif isinstance(content_data, str):
                    return content_data
        
        return ""
    
    def _extract_tags_from_entry(self, entry) -> List[str]:
        """Extract tags from RSS entry"""
        tags = []
        
        if hasattr(entry, 'tags'):
            for tag in entry.tags:
                if hasattr(tag, 'term'):
                    tags.append(tag.term)
        
        if hasattr(entry, 'categories'):
            tags.extend(entry.categories)
        
        return tags
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_full_content(self, client: httpx.AsyncClient, url: str) -> Optional[str]:
        """Fetch full HTML content from URL and convert to markdown"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
            
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            # Convert HTML to markdown
            markdown_content = self.h2t.handle(response.text)
            
            # Clean up the markdown
            cleaned_content = self._clean_markdown(markdown_content)
            
            return cleaned_content
            
        except Exception as e:
            logger.warning(f"Failed to fetch full content from {url}: {e}")
            return None
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean up converted markdown content"""
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown)
        
        # Remove navigation elements and footers
        lines = cleaned.split('\n')
        filtered_lines = []
        
        skip_patterns = [
            r'^\s*\*\s*(home|about|contact|privacy|terms)', 
            r'^\s*©', 
            r'^\s*all rights reserved',
            r'^\s*subscribe',
            r'^\s*follow us',
            r'^\s*share this',
            r'^\s*\[.*\]\(#.*\)$'  # Skip internal anchor links
        ]
        
        for line in lines:
            line_lower = line.lower()
            if not any(re.match(pattern, line_lower) for pattern in skip_patterns):
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines).strip()
    
    def is_postmortem_content(self, title: str, content: str, summary: str = "") -> bool:
        """Check if content appears to be a postmortem using existing logic"""
        combined_text = f"{title} {content} {summary}".lower()
        
        # Must contain postmortem keywords
        has_postmortem_keywords = any(keyword in combined_text for keyword in self.postmortem_keywords)
        
        # Must have sufficient content
        has_sufficient_content = len(content) >= self.min_content_length
        
        # Must mention technical/infrastructure terms
        tech_terms = [
            "service", "system", "infrastructure", "database", "server", "api",
            "application", "network", "deployment", "monitoring", "load balancer",
            "kubernetes", "docker", "aws", "gcp", "azure", "redis", "nginx"
        ]
        has_tech_content = any(term in combined_text for term in tech_terms)
        
        return has_postmortem_keywords and has_sufficient_content and has_tech_content
    
    async def harvest_rss_feeds(self, max_posts_per_source: int = 50) -> List[PostmortemData]:
        """Harvest postmortem content from all RSS feeds"""
        all_postmortems = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for source_name, feed_urls in self.rss_feeds.items():
                logger.info(f"🔍 Processing {source_name} feeds...")
                source_postmortems = []
                
                for feed_url in feed_urls:
                    try:
                        # Fetch RSS items
                        rss_items = await self.fetch_rss_feed(feed_url)
                        
                        for item in tqdm(rss_items, desc=f"Processing {source_name}"):
                            title = item.get("title", "")
                            link = item.get("link", "")
                            summary = item.get("summary", "")
                            initial_content = item.get("content", "")
                            
                            if not link:
                                continue
                            
                            # Fetch full content from the link
                            full_content = await self.fetch_full_content(client, link)
                            if not full_content:
                                # Fallback to RSS content
                                full_content = initial_content or summary
                            
                            # Check if it's a postmortem
                            if self.is_postmortem_content(title, full_content, summary):
                                # Extract metadata using existing postmortem logic
                                metadata = self.pm_harvester.extract_postmortem_metadata(
                                    full_content, link, title
                                )
                                
                                # Apply quality filter
                                if metadata["quality_score"] >= self.min_quality_score:
                                    postmortem = PostmortemData(
                                        source=f"rss:{source_name}",
                                        url=link,
                                        title=title,
                                        content=full_content,
                                        timestamp=item.get("published", datetime.now().isoformat()),
                                        tags=metadata["tags"] + item.get("tags", []),
                                        severity=metadata["severity"],
                                        services_affected=metadata["services_affected"],
                                        infrastructure_components=metadata["infrastructure_components"],
                                        failure_pattern=metadata["failure_pattern"],
                                        timeline_events=metadata["timeline_events"],
                                        blast_radius=metadata["blast_radius"],
                                        mitigation_actions=metadata["mitigation_actions"],
                                        quality_score=metadata["quality_score"]
                                    )
                                    source_postmortems.append(postmortem)
                                    
                                    logger.debug(f"✅ Added {source_name} postmortem: {title[:60]}...")
                            
                            # Rate limiting
                            await asyncio.sleep(1.0)
                            
                            if len(source_postmortems) >= max_posts_per_source:
                                break
                        
                        if len(source_postmortems) >= max_posts_per_source:
                            break
                            
                    except Exception as e:
                        logger.error(f"❌ Error processing feed {feed_url}: {e}")
                        continue
                
                all_postmortems.extend(source_postmortems)
                logger.info(f"✅ {source_name}: {len(source_postmortems)} postmortems found")
        
        return all_postmortems
    
    def save_postmortems(self, postmortems: List[PostmortemData], filename: str = None) -> str:
        """Save harvested postmortems to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"incidentio_rss_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        data = [asdict(pm) for pm in postmortems]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Also save as parquet
        if data:
            df = pd.DataFrame(data)
            parquet_path = output_path.with_suffix('.parquet')
            df.to_parquet(parquet_path, index=False, engine='fastparquet')
        
        logger.info(f"💾 Saved {len(postmortems)} RSS postmortems to {output_path}")
        return str(output_path)


async def main():
    """Main harvesting function for RSS feeds"""
    from dotenv import load_dotenv
    load_dotenv()
    
    output_dir = os.getenv("DATA_DIR", "data/raw") + "/postmortems"
    max_posts = 50  # Target: ~50 docs as specified
    
    harvester = StatusRSSHarvester(output_dir)
    
    try:
        logger.info("🚀 Starting RSS status feed harvest...")
        start_time = time.time()
        
        postmortems = await harvester.harvest_rss_feeds(max_posts_per_source=max_posts)
        
        if postmortems:
            output_file = harvester.save_postmortems(postmortems)
            
            # Generate summary
            if postmortems:
                df = pd.DataFrame([asdict(pm) for pm in postmortems])
                
                logger.info("📊 RSS Harvest Summary:")
                logger.info(f"   Total postmortems: {len(postmortems)}")
                logger.info(f"   Sources: {df['source'].nunique()}")
                logger.info(f"   Avg quality: {df['quality_score'].mean():.2f}")
                logger.info(f"   High quality (≥0.5): {len(df[df['quality_score'] >= 0.5])}")
                logger.info(f"   Failure patterns: {df['failure_pattern'].value_counts().head(3).to_dict()}")
                logger.info(f"   Output file: {output_file}")
        else:
            logger.warning("❌ No quality postmortems found in RSS feeds")
        
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️  RSS harvest completed in {elapsed_time:.1f}s")
        
    except Exception as e:
        logger.error(f"❌ RSS harvest failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 