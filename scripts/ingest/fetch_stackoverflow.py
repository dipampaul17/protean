#!/usr/bin/env python3
"""
Fetch Stack Overflow Infrastructure Q&A for Pattern Analysis
Part of Protean Pattern Discovery Engine - Enhanced with API Key for Higher Quota
"""

import os
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import gzip

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from loguru import logger


@dataclass
class StackOverflowPost:
    """Structure for Stack Overflow post data"""
    post_id: int
    post_type: str  # question or answer
    title: Optional[str]
    body: str
    tags: List[str]
    score: int
    view_count: Optional[int]
    answer_count: Optional[int]
    favorite_count: Optional[int]
    creation_date: str
    last_activity_date: str
    owner_reputation: Optional[int]
    is_answered: Optional[bool]
    accepted_answer_id: Optional[int]
    link: str
    infrastructure_components: List[str]
    problem_category: Optional[str]
    quality_score: float = 0.0  # ENHANCEMENT 5: Add quality score


class StackOverflowHarvester:
    """Harvest infrastructure-related content from Stack Overflow with API key"""
    
    def __init__(self, output_dir: str = "data/raw/stackoverflow"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Base API URL
        self.api_base = "https://api.stackexchange.com/2.3"
        
        # API credentials for higher quota
        self.api_key = "rl_HLLQJAcsHeoqPKn3hZnmFr1sw"  # 10,000 requests/day with key vs 300 without
        self.client_id = "33280"
        
        # Enhanced request tracking
        self.request_count = 0
        self.quota_remaining = 10000
        self.backoff_until = None
        
        # Infrastructure-related tags to search for - optimized for high-quality content
        self.infra_tags = [
            # High-volume, high-quality tags (most valuable for patterns)
            "docker", "kubernetes", "nginx", "postgresql", "redis", "elasticsearch",
            "amazon-web-services", "terraform", "jenkins", "monitoring",
            "docker-compose", "mysql", "mongodb", "apache-kafka", "rabbitmq",
            "prometheus", "grafana", "ansible", "performance", "scaling"
        ]
        
        # Problem categories for classification
        self.problem_categories = {
            "performance": ["slow", "performance", "latency", "timeout", "bottleneck", "optimization"],
            "scaling": ["scale", "scaling", "load", "capacity", "traffic", "horizontal", "vertical"],
            "availability": ["down", "outage", "unavailable", "crash", "failure", "reliability"],
            "security": ["security", "vulnerability", "attack", "breach", "auth", "ssl", "encryption"],
            "configuration": ["config", "setup", "install", "deploy", "configure", "environment"],
            "networking": ["network", "connection", "dns", "firewall", "port", "proxy", "routing"],
            "storage": ["storage", "disk", "volume", "backup", "restore", "persistence", "database"],
            "monitoring": ["monitor", "alert", "log", "metric", "dashboard", "observability", "tracing"]
        }
        
        # Enhanced quality thresholds
        self.min_score_threshold = 2  # Minimum score for questions
        self.min_answer_length = 800  # ENHANCEMENT 5: Tightened from 400 to 800
        self.min_quality_score = 0.4  # ENHANCEMENT 5: Tightened from 0.2 to 0.4
    
    def check_api_quota(self, response_headers: Dict[str, str]):
        """Monitor API quota and handle rate limiting"""
        if 'x-ratelimit-remaining' in response_headers:
            self.quota_remaining = int(response_headers['x-ratelimit-remaining'])
            logger.debug(f"API quota remaining: {self.quota_remaining}")
            
        # Handle backoff
        if 'x-ratelimit-backoff' in response_headers:
            backoff_seconds = int(response_headers['x-ratelimit-backoff'])
            self.backoff_until = time.time() + backoff_seconds
            logger.warning(f"API backoff requested: {backoff_seconds}s")
            
        # Log quota warnings
        if self.quota_remaining < 100:
            logger.warning(f"⚠️  Low API quota remaining: {self.quota_remaining}")
    
    async def wait_for_backoff(self):
        """Wait if we're in backoff period"""
        if self.backoff_until and time.time() < self.backoff_until:
            wait_time = self.backoff_until - time.time()
            logger.info(f"⏱️  Waiting for API backoff: {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            self.backoff_until = None
    
    def extract_infrastructure_components(self, text: str, tags: List[str]) -> List[str]:
        """Extract infrastructure components from text and tags"""
        components = set()
        
        # Add tags that match infrastructure components
        for tag in tags:
            if any(infra_tag in tag for infra_tag in self.infra_tags):
                components.add(tag)
        
        # Extract from text content with enhanced patterns
        text_lower = text.lower()
        infra_keywords = {
            "containers": ["kubernetes", "k8s", "docker", "podman", "containerd"],
            "web_servers": ["nginx", "apache", "httpd", "haproxy", "traefik"],
            "databases": ["postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra"],
            "queues": ["kafka", "rabbitmq", "sqs", "pubsub", "nats"],
            "cloud": ["aws", "gcp", "azure", "terraform", "cloudformation"],
            "monitoring": ["prometheus", "grafana", "datadog", "newrelic", "elk"],
            "cicd": ["jenkins", "gitlab", "github-actions", "ansible", "puppet"]
        }
        
        for category, keywords in infra_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    components.add(keyword)
        
        return list(components)
    
    def classify_problem_category(self, title: str, body: str) -> Optional[str]:
        """Classify the problem category based on content"""
        content = (title + " " + body).lower()
        
        # Score each category
        category_scores = {}
        for category, keywords in self.problem_categories.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score > 0:
                category_scores[category] = score
        
        # Return highest scoring category
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return "general"
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=30))
    async def fetch_questions_by_tag(self, client: httpx.AsyncClient, tag: str, 
                                   page_size: int = 100, max_pages: int = 3) -> List[Dict[str, Any]]:
        """Fetch questions for a specific tag with enhanced error handling"""
        all_questions = []
        
        for page in range(1, max_pages + 1):
            # Wait for backoff if needed
            await self.wait_for_backoff()
            
            params = {
                "site": "stackoverflow",
                "tagged": tag,
                "sort": "votes",  # Get highest quality content first
                "order": "desc",
                "pagesize": page_size,
                "page": page,
                "filter": "withbody",  # Valid filter that includes body content
                "key": self.api_key,  # API key for higher quota
                "min": 5  # Minimum score to filter quality
            }
            
            try:
                logger.debug(f"Fetching {tag} questions, page {page}")
                response = await client.get(f"{self.api_base}/questions", params=params)
                
                # Check for API errors
                if response.status_code == 429:
                    logger.warning(f"Rate limited on tag {tag}, page {page}")
                    await asyncio.sleep(5)  # Wait longer on rate limit
                    continue
                elif response.status_code == 400:
                    logger.error(f"Bad request for tag {tag}: {response.text}")
                    break
                
                response.raise_for_status()
                data = response.json()
                
                # Check API quota
                self.check_api_quota(response.headers)
                self.request_count += 1
                
                # Handle API errors in response
                if "error_id" in data:
                    logger.error(f"API error for tag {tag}: {data.get('error_message', 'Unknown error')}")
                    break
                
                questions = data.get("items", [])
                if not questions:
                    logger.debug(f"No more questions for tag {tag}")
                    break
                    
                all_questions.extend(questions)
                logger.debug(f"Fetched {len(questions)} questions for tag {tag}, page {page}")
                
                # Check if we've reached the end
                if not data.get("has_more", False):
                    break
                    
                # Enhanced rate limiting with API key (higher quota)
                await asyncio.sleep(0.1)  # Reduced from 0.5s due to higher quota
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching questions for tag {tag}, page {page}: {e}")
                if e.response.status_code == 429:
                    await asyncio.sleep(10)  # Longer wait on rate limit
                    continue
                else:
                    break
            except Exception as e:
                logger.error(f"Unexpected error fetching questions for tag {tag}, page {page}: {e}")
                break
        
        logger.info(f"✅ Tag {tag}: {len(all_questions)} questions fetched")
        return all_questions
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=30))
    async def fetch_answers_for_question(self, client: httpx.AsyncClient, question_id: int) -> List[Dict[str, Any]]:
        """Fetch answers for a specific question with enhanced error handling"""
        # Wait for backoff if needed
        await self.wait_for_backoff()
        
        params = {
            "site": "stackoverflow",
            "sort": "votes",
            "order": "desc",
            "filter": "withbody",  # Valid filter for answer data
            "key": self.api_key,
            "pagesize": 10  # Top 10 answers maximum
        }
        
        try:
            response = await client.get(f"{self.api_base}/questions/{question_id}/answers", params=params)
            
            if response.status_code == 429:
                logger.warning(f"Rate limited fetching answers for question {question_id}")
                await asyncio.sleep(5)
                return []
            elif response.status_code == 400:
                logger.error(f"Bad request for question {question_id} answers: {response.text}")
                return []
                
            response.raise_for_status()
            data = response.json()
            
            # Check API quota
            self.check_api_quota(response.headers)
            self.request_count += 1
            
            # Handle API errors
            if "error_id" in data:
                logger.error(f"API error for question {question_id} answers: {data.get('error_message', 'Unknown error')}")
                return []
            
            return data.get("items", [])
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching answers for question {question_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching answers for question {question_id}: {e}")
            return []
    
    def calculate_quality_score(self, item: Dict[str, Any], post_type: str) -> float:
        """Enhanced quality score calculation"""
        score = 0.0
        
        # Score based metrics (weighted higher)
        post_score = item.get("score", 0)
        if post_score >= 20:
            score += 0.4
        elif post_score >= 10:
            score += 0.3
        elif post_score >= 5:
            score += 0.2
        elif post_score >= 1:
            score += 0.1
        
        # Content length (adjusted for quality)
        body_length = len(item.get("body", ""))
        if body_length > 2000:
            score += 0.3
        elif body_length > 1200:
            score += 0.25
        elif body_length > 800:
            score += 0.2
        elif body_length > 400:
            score += 0.1
        
        # Accepted answer bonus (strong quality indicator)
        if post_type == "answer" and item.get("is_accepted", False):
            score += 0.3
        
        # Answer count for questions (engagement indicator)
        if post_type == "question":
            answer_count = item.get("answer_count", 0)
            if answer_count >= 10:
                score += 0.2
            elif answer_count >= 5:
                score += 0.15
            elif answer_count >= 2:
                score += 0.1
        
        # Owner reputation (reliability indicator)
        owner_rep = item.get("owner", {}).get("reputation", 0)
        if owner_rep >= 50000:
            score += 0.15
        elif owner_rep >= 10000:
            score += 0.1
        elif owner_rep >= 1000:
            score += 0.05
        
        # View count for questions (popularity indicator)
        if post_type == "question":
            view_count = item.get("view_count", 0)
            if view_count >= 10000:
                score += 0.1
            elif view_count >= 1000:
                score += 0.05
        
        return min(score, 1.0)

    def convert_to_post_object(self, item: Dict[str, Any], post_type: str = "question") -> StackOverflowPost:
        """Convert API response to StackOverflowPost object"""
        tags = item.get("tags", [])
        title = item.get("title", "")
        body = item.get("body", "")
        
        # Extract infrastructure components
        infra_components = self.extract_infrastructure_components(title + " " + body, tags)
        
        # Classify problem category
        problem_category = self.classify_problem_category(title, body)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(item, post_type)
        
        return StackOverflowPost(
            post_id=item["question_id"] if post_type == "question" else item.get("answer_id", 0),
            post_type=post_type,
            title=title if post_type == "question" else None,
            body=body,
            tags=tags,
            score=item.get("score", 0),
            view_count=item.get("view_count"),
            answer_count=item.get("answer_count"),
            favorite_count=item.get("favorite_count"),
            creation_date=datetime.fromtimestamp(item["creation_date"]).isoformat(),
            last_activity_date=datetime.fromtimestamp(item["last_activity_date"]).isoformat(),
            owner_reputation=item.get("owner", {}).get("reputation"),
            is_answered=item.get("is_answered"),
            accepted_answer_id=item.get("accepted_answer_id"),
            link=item.get("link", f"https://stackoverflow.com/questions/{item.get('question_id', 'unknown')}"),
            infrastructure_components=infra_components,
            problem_category=problem_category,
            quality_score=quality_score
        )
    
    async def harvest_by_tags(self, max_posts_per_tag: int = 200) -> List[StackOverflowPost]:
        """Harvest posts by infrastructure tags with enhanced filtering"""
        all_posts = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout
            for tag in tqdm(self.infra_tags, desc="🔍 Harvesting SO tags"):
                try:
                    # Check quota before proceeding
                    if self.quota_remaining < 10:
                        logger.warning("⚠️  API quota too low, stopping tag harvest")
                        break
                    
                    # Fetch questions for this tag
                    questions = await self.fetch_questions_by_tag(
                        client, tag, 
                        page_size=100, 
                        max_pages=max(1, max_posts_per_tag // 100)
                    )
                    
                    tag_posts = []
                    for question in questions:
                        # Convert question to post object and apply quality filter
                        question_post = self.convert_to_post_object(question, "question")
                        
                        # Higher quality threshold for questions
                        if (question_post.score >= self.min_score_threshold and 
                            question_post.quality_score >= 0.3):
                            tag_posts.append(question_post)
                        
                        # Fetch answers for high-quality questions
                        if (question.get("score", 0) >= 5 and 
                            question.get("answer_count", 0) > 0 and
                            len(tag_posts) < max_posts_per_tag):
                            
                            answers = await self.fetch_answers_for_question(client, question["question_id"])
                            
                            for answer in answers[:3]:  # Top 3 answers only
                                answer_post = self.convert_to_post_object(answer, "answer")
                                
                                # ENHANCEMENT 5: Tightened filter - len(answer) > 800 and quality ≥0.4
                                if (len(answer.get("body", "")) > self.min_answer_length and 
                                    answer_post.quality_score >= self.min_quality_score):
                                    tag_posts.append(answer_post)
                            
                            # Rate limiting between answer requests
                            await asyncio.sleep(0.1)
                    
                    all_posts.extend(tag_posts)
                    logger.info(f"✅ Tag '{tag}': {len(tag_posts)} high-quality posts")
                    
                    # More aggressive rate limiting if quota is low
                    if self.quota_remaining < 100:
                        await asyncio.sleep(1.0)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing tag '{tag}': {e}")
                    continue
        
        logger.info(f"✅ Total SO posts harvested: {len(all_posts)}")
        return all_posts
    
    async def search_by_keywords(self, keywords: List[str], max_results: int = 500) -> List[StackOverflowPost]:
        """Search posts by infrastructure keywords with enhanced error handling"""
        all_posts = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for keyword in tqdm(keywords, desc="🔍 Searching SO keywords"):
                if self.quota_remaining < 10:
                    logger.warning("⚠️  API quota too low, stopping keyword search")
                    break
                
                await self.wait_for_backoff()
                
                params = {
                    "site": "stackoverflow", 
                    "intitle": keyword,
                    "sort": "votes",
                    "order": "desc",
                    "pagesize": min(100, max_results // len(keywords)),
                    "filter": "withbody",
                    "key": self.api_key,
                    "min": 3  # Minimum score filter
                }
                
                try:
                    response = await client.get(f"{self.api_base}/search", params=params)
                    
                    if response.status_code == 429:
                        logger.warning(f"Rate limited searching for keyword '{keyword}'")
                        await asyncio.sleep(5)
                        continue
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    self.check_api_quota(response.headers)
                    self.request_count += 1
                    
                    keyword_posts = []
                    for item in data.get("items", []):
                        post = self.convert_to_post_object(item, "question")
                        if post.quality_score >= 0.3:  # Quality filter
                            keyword_posts.append(post)
                    
                    all_posts.extend(keyword_posts)
                    logger.info(f"✅ Keyword '{keyword}': {len(keyword_posts)} posts")
                    
                    await asyncio.sleep(0.2)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"❌ Error searching for keyword '{keyword}': {e}")
                    continue
        
        return all_posts
    
    def save_posts(self, posts: List[StackOverflowPost], filename: str = None) -> str:
        """Save harvested posts to file with enhanced metadata"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stackoverflow_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        data = [asdict(post) for post in posts]
        
        # Save as compressed JSON
        with gzip.open(str(output_path) + ".gz", 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Also save as parquet for faster processing
        if data:
            df = pd.DataFrame(data)
            parquet_path = output_path.with_suffix('.parquet')
            df.to_parquet(parquet_path, index=False, engine='fastparquet')
        
        # Save harvest metadata
        metadata = {
            "total_posts": len(posts),
            "api_requests_made": self.request_count,
            "quota_remaining": self.quota_remaining,
            "harvest_timestamp": datetime.now().isoformat(),
            "quality_thresholds": {
                "min_score": self.min_score_threshold,
                "min_answer_length": self.min_answer_length,
                "min_quality_score": self.min_quality_score
            }
        }
        
        metadata_path = output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Saved {len(posts)} posts to {output_path}.gz")
        logger.info(f"📊 API requests used: {self.request_count}, quota remaining: {self.quota_remaining}")
        return str(output_path)
    
    async def harvest_all(self, max_posts: int = 2000) -> List[StackOverflowPost]:
        """Enhanced harvest with API key and better error handling"""
        logger.info("🔍 Starting Stack Overflow harvest with API key")
        logger.info(f"📊 Target: {max_posts} high-quality posts")
        
        all_posts = []
        
        # Harvest by tags (primary method - higher volume)
        logger.info("🏷️  Harvesting by infrastructure tags...")
        tag_posts = await self.harvest_by_tags(max_posts_per_tag=max_posts // len(self.infra_tags))
        all_posts.extend(tag_posts)
        
        # Search by keywords for additional coverage (if quota allows)
        if self.quota_remaining > 100:
            search_keywords = [
                "infrastructure failure", "system outage", "performance issue", 
                "deployment problem", "scaling issue", "monitoring setup",
                "database performance", "load balancer", "kubernetes error"
            ]
            logger.info("🔍 Searching by infrastructure keywords...")
            keyword_posts = await self.search_by_keywords(search_keywords, max_results=500)
            all_posts.extend(keyword_posts)
        else:
            logger.warning("⚠️  Skipping keyword search due to low quota")
        
        # Deduplicate based on post_id
        unique_posts = []
        seen_ids = set()
        for post in all_posts:
            if post.post_id not in seen_ids:
                unique_posts.append(post)
                seen_ids.add(post.post_id)
        
        # Sort by quality score (highest first)
        unique_posts.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Log quality statistics
        if unique_posts:
            quality_scores = [p.quality_score for p in unique_posts]
            logger.info(f"✅ Quality stats - Avg: {sum(quality_scores)/len(quality_scores):.2f}, "
                       f"Min: {min(quality_scores):.2f}, Max: {max(quality_scores):.2f}")
            
            # Count by category
            categories = {}
            for post in unique_posts:
                cat = post.problem_category or "unknown"
                categories[cat] = categories.get(cat, 0) + 1
            logger.info(f"📊 Categories: {categories}")
        
        logger.info(f"✅ Harvested {len(unique_posts)} unique Stack Overflow posts")
        return unique_posts


async def main():
    """Main harvesting function with enhanced error tracking"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    max_posts = int(os.getenv("MAX_STACKOVERFLOW_POSTS", 500))  # Realistic target for testing
    output_dir = os.getenv("DATA_DIR", "data/raw") + "/stackoverflow"
    
    # Initialize harvester
    harvester = StackOverflowHarvester(output_dir)
    
    # Set time limit for harvesting
    max_time = int(os.getenv("MAX_HARVEST_TIME", 7200))  # 2 hours default
    start_time = time.time()
    
    try:
        # Harvest posts
        posts = await harvester.harvest_all(max_posts)
        
        # Save results
        output_file = harvester.save_posts(posts)
        
        # Generate enhanced summary statistics
        if posts:
            df = pd.DataFrame([asdict(post) for post in posts])
            
            logger.info("📊 Enhanced Harvest Summary:")
            logger.info(f"   Total posts: {len(posts)}")
            logger.info(f"   Questions: {len(df[df['post_type'] == 'question'])}")
            logger.info(f"   Answers: {len(df[df['post_type'] == 'answer'])}")
            logger.info(f"   Avg score: {df['score'].mean():.1f}")
            logger.info(f"   Avg quality: {df['quality_score'].mean():.2f}")
            logger.info(f"   High quality (≥0.5): {len(df[df['quality_score'] >= 0.5])}")
            logger.info(f"   API requests: {harvester.request_count}")
            logger.info(f"   Quota remaining: {harvester.quota_remaining}")
            logger.info(f"   Top tags: {pd.Series([tag for tags in df['tags'] for tag in tags]).value_counts().head(5).to_dict()}")
            logger.info(f"   Problem categories: {df['problem_category'].value_counts().to_dict()}")
            logger.info(f"   Output file: {output_file}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️  Harvest completed in {elapsed_time:.1f}s")
        
    except Exception as e:
        logger.error(f"❌ Harvest failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
