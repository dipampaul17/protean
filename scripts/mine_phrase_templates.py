#!/usr/bin/env python3
"""
Mine Phrase Templates for Synthetic Data Generation
Part of Protean Pattern Discovery Engine - Week 1 Mission

Read documents WHERE length(text)>300 AND pattern_hint IS NOT NULL FROM corpus.db
Use basic NLP techniques for phrase extraction
Keep phrases 2–6 tokens, lowercase, frequency>3
Write YAML: {pattern_hint:[phrase1, phrase2, ...]}
↳ data/synthetic/phrase_templates.yaml
"""

import sqlite3
import re
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
from datetime import datetime
import yaml

from loguru import logger
from tqdm import tqdm


class SimplePhraseExtractor:
    """Extract meaningful phrases using basic NLP techniques"""
    
    def __init__(self):
        self.infrastructure_terms = self._load_infrastructure_terms()
        self.stop_words = self._load_stop_words()
        
    def _load_infrastructure_terms(self) -> Set[str]:
        """Load infrastructure-specific terms to prioritize"""
        return {
            # Core infrastructure
            'server', 'database', 'cache', 'queue', 'cluster', 'node',
            'container', 'pod', 'service', 'endpoint', 'api', 'gateway',
            'load balancer', 'proxy', 'firewall', 'network', 'storage',
            
            # Cloud services
            'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'redis',
            'mysql', 'postgres', 'mongodb', 'elasticsearch', 'kafka',
            'rabbitmq', 'nginx', 'apache', 'tomcat', 'jenkins',
            
            # Failure modes
            'timeout', 'latency', 'error', 'failure', 'outage', 'downtime',
            'crash', 'hang', 'stuck', 'slow', 'degraded', 'unavailable',
            'connection', 'memory', 'cpu', 'disk', 'capacity', 'limit',
            
            # Operations
            'deploy', 'deployment', 'release', 'rollback', 'restart',
            'scale', 'monitor', 'alert', 'log', 'debug', 'troubleshoot',
            'diagnose', 'investigate', 'resolve', 'fix', 'patch', 'update'
        }
    
    def _load_stop_words(self) -> Set[str]:
        """Load stop words for filtering"""
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
            'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
            "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
            'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
            'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', 
            "wouldn't", 'said', 'say', 'says', 'told', 'tell', 'asked', 'ask', 'think', 'thought', 
            'know', 'knew', 'see', 'saw', 'get', 'got', 'put', 'take', 'took', 'make', 'made', 
            'go', 'went', 'come', 'came', 'want', 'wanted', 'like', 'time', 'way', 'day', 'year', 
            'good', 'new'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs, emails, and special characters
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases using simple patterns"""
        phrases = []
        
        # Simple noun phrase patterns
        # Pattern 1: adjective + noun combinations
        adj_noun_pattern = r'\b([a-z]+(?:ed|ing|ive|ful|less|able)\s+[a-z]+)\b'
        matches = re.finditer(adj_noun_pattern, text, re.IGNORECASE)
        for match in matches:
            phrase = match.group(1).lower().strip()
            if self._is_valid_phrase(phrase):
                phrases.append(phrase)
        
        # Pattern 2: noun + noun combinations
        noun_noun_pattern = r'\b([a-z]+\s+(?:server|database|service|system|application|error|failure|timeout|connection|network|cluster|node|container|pod|deployment|cache|queue|storage|memory|cpu|disk|capacity|limit|monitor|alert|log|debug|scale|deploy|restart|rollback|update|patch|fix|resolve))\b'
        matches = re.finditer(noun_noun_pattern, text, re.IGNORECASE)
        for match in matches:
            phrase = match.group(1).lower().strip()
            if self._is_valid_phrase(phrase):
                phrases.append(phrase)
        
        # Pattern 3: Infrastructure term combinations
        for term in self.infrastructure_terms:
            if ' ' in term and term in text.lower():
                phrases.append(term)
        
        return phrases
    
    def extract_verb_phrases(self, text: str) -> List[str]:
        """Extract verb phrases using simple patterns"""
        phrases = []
        
        # Pattern 1: verb + object combinations for infrastructure actions
        verb_patterns = [
            r'\b(restart(?:ed|ing)?\s+(?:server|service|application|container|pod|database))\b',
            r'\b(deploy(?:ed|ing)?\s+(?:application|service|container|update|patch))\b',
            r'\b(scale(?:d|ing)?\s+(?:up|down|out|in)\s+(?:server|service|application|cluster))\b',
            r'\b(monitor(?:ed|ing)?\s+(?:performance|metrics|logs|alerts|system|application))\b',
            r'\b(debug(?:ged|ging)?\s+(?:issue|problem|error|failure|application|service))\b',
            r'\b(troubleshoot(?:ed|ing)?\s+(?:issue|problem|error|network|connection|service))\b',
            r'\b(resolve(?:d|ing)?\s+(?:issue|problem|error|failure|incident|outage))\b',
            r'\b(investigate(?:d|ing)?\s+(?:issue|problem|error|failure|incident|cause|root))\b',
            r'\b(fix(?:ed|ing)?\s+(?:issue|problem|error|bug|vulnerability|configuration))\b',
            r'\b(update(?:d|ing)?\s+(?:configuration|settings|version|application|service))\b',
            r'\b(patch(?:ed|ing)?\s+(?:system|application|service|vulnerability|security))\b',
            r'\b(rollback(?:ed|ing)?\s+(?:deployment|update|change|version|release))\b'
        ]
        
        for pattern in verb_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phrase = match.group(1).lower().strip()
                if self._is_valid_phrase(phrase):
                    phrases.append(phrase)
        
        return phrases
    
    def extract_technical_patterns(self, text: str) -> List[str]:
        """Extract technical patterns specific to infrastructure"""
        patterns = []
        
        # Error patterns
        error_patterns = [
            r'(timeout|timed out) after \d+',
            r'connection (refused|reset|lost)',
            r'(memory|cpu|disk) (usage|utilization) (above|over|exceeded) \d+',
            r'(failed to|unable to|could not) (connect|reach|access)',
            r'(high|increased|elevated) (latency|response time)',
            r'(service|server|database) (unavailable|unreachable)',
            r'(circuit breaker|rate limit) (triggered|activated)',
            r'(load balancer|proxy) (error|failure)',
            r'(deployment|release) (failed|rolled back)',
            r'(scale|scaling) (up|down|out) (failed|issues)'
        ]
        
        for pattern in error_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                phrase = match.group(0)
                tokens = phrase.split()
                if 2 <= len(tokens) <= 6:
                    patterns.append(phrase)
        
        return patterns
    
    def _is_valid_phrase(self, phrase: str) -> bool:
        """Check if a phrase is valid for extraction"""
        tokens = phrase.split()
        
        # Filter by length
        if not (2 <= len(tokens) <= 6):
            return False
        
        # Skip if all tokens are stop words
        if all(token in self.stop_words for token in tokens):
            return False
        
        # Skip if phrase is too short
        if len(phrase) < 4:
            return False
        
        # Must contain at least one infrastructure-related term or be technical
        has_infra_term = any(term in phrase for term in self.infrastructure_terms)
        has_technical_word = any(word in phrase for word in ['error', 'fail', 'timeout', 'crash', 'slow', 'down', 'issue', 'problem'])
        
        return has_infra_term or has_technical_word
    
    def extract_phrases_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract all types of phrases from text"""
        # Clean and process text
        cleaned_text = self.clean_text(text)
        
        # Limit text length for efficiency
        if len(cleaned_text) > 10000:
            cleaned_text = cleaned_text[:10000]
        
        phrases = {
            'noun_phrases': self.extract_noun_phrases(cleaned_text),
            'verb_phrases': self.extract_verb_phrases(cleaned_text),
            'technical_patterns': self.extract_technical_patterns(cleaned_text)
        }
        
        return phrases


class PhraseMiner:
    """Main phrase mining orchestrator"""
    
    def __init__(self, db_path: str = "corpus.db", output_dir: str = "data/synthetic"):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = SimplePhraseExtractor()
        self.phrase_counter = defaultdict(lambda: defaultdict(Counter))
        self.min_frequency = 3
        self.min_text_length = 300
    
    def load_documents(self) -> List[Tuple[str, str]]:
        """Load documents from corpus database"""
        if not Path(self.db_path).exists():
            logger.error(f"❌ Database not found: {self.db_path}")
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query documents with pattern hints and sufficient length
        query = """
        SELECT pattern_hint, text 
        FROM documents 
        WHERE length(text) > ? 
        AND pattern_hint IS NOT NULL 
        AND pattern_hint != ''
        ORDER BY pattern_hint, length(text) DESC
        """
        
        cursor.execute(query, (self.min_text_length,))
        documents = cursor.fetchall()
        conn.close()
        
        logger.info(f"📄 Loaded {len(documents)} documents for phrase mining")
        return documents
    
    def mine_phrases(self, documents: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Mine phrases from documents grouped by pattern"""
        logger.info("🔍 Mining phrases from documents...")
        
        # Process documents
        for pattern_hint, text in tqdm(documents, desc="Processing documents"):
            try:
                phrases = self.extractor.extract_phrases_from_text(text)
                
                # Count phrases by type
                for phrase_type, phrase_list in phrases.items():
                    for phrase in phrase_list:
                        self.phrase_counter[pattern_hint][phrase_type][phrase] += 1
            
            except Exception as e:
                logger.warning(f"Error processing document for {pattern_hint}: {e}")
                continue
        
        # Filter phrases by frequency and create final templates
        phrase_templates = defaultdict(list)
        
        for pattern_hint, phrase_types in self.phrase_counter.items():
            all_phrases = []
            
            # Combine all phrase types
            for phrase_type, phrase_counts in phrase_types.items():
                # Filter by frequency
                frequent_phrases = [
                    phrase for phrase, count in phrase_counts.items() 
                    if count >= self.min_frequency
                ]
                all_phrases.extend(frequent_phrases)
            
            # Remove duplicates and sort by estimated importance
            unique_phrases = list(set(all_phrases))
            
            # Score phrases (infrastructure terms get higher scores)
            def phrase_score(phrase):
                score = 0
                words = phrase.split()
                
                # Infrastructure term bonus
                for word in words:
                    if word in self.extractor.infrastructure_terms:
                        score += 10
                
                # Length bonus (3-4 words are often most useful)
                if 3 <= len(words) <= 4:
                    score += 5
                
                # Total frequency across all phrase types
                total_freq = sum(
                    phrase_counts.get(phrase, 0) 
                    for phrase_counts in phrase_types.values()
                )
                score += total_freq
                
                return score
            
            # Sort by score and take top phrases
            scored_phrases = sorted(unique_phrases, key=phrase_score, reverse=True)
            phrase_templates[pattern_hint] = scored_phrases[:50]  # Limit to top 50 per pattern
        
        return dict(phrase_templates)
    
    def save_phrase_templates(self, phrase_templates: Dict[str, List[str]]) -> str:
        """Save phrase templates to YAML file"""
        output_file = self.output_dir / "phrase_templates.yaml"
        
        # Add metadata
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_patterns': len(phrase_templates),
                'total_phrases': sum(len(phrases) for phrases in phrase_templates.values()),
                'min_frequency': self.min_frequency,
                'min_text_length': self.min_text_length,
                'description': 'Infrastructure failure phrase templates mined from corpus'
            },
            'phrase_templates': phrase_templates
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=True, indent=2)
        
        logger.info(f"💾 Saved phrase templates to {output_file}")
        return str(output_file)
    
    def generate_statistics(self, phrase_templates: Dict[str, List[str]]) -> Dict[str, any]:
        """Generate mining statistics"""
        stats = {
            'total_patterns': len(phrase_templates),
            'total_phrases': sum(len(phrases) for phrases in phrase_templates.values()),
            'phrases_per_pattern': {
                pattern: len(phrases) 
                for pattern, phrases in phrase_templates.items()
            },
            'top_patterns_by_phrase_count': sorted(
                phrase_templates.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )[:10]
        }
        
        return stats
    
    def run_mining(self) -> Dict[str, any]:
        """Main mining workflow"""
        start_time = datetime.now()
        logger.info("🚀 Starting phrase template mining...")
        
        # Load documents
        documents = self.load_documents()
        if not documents:
            logger.error("❌ No documents found for mining")
            return {"success": False}
        
        # Mine phrases
        phrase_templates = self.mine_phrases(documents)
        
        if not phrase_templates:
            logger.error("❌ No phrases extracted")
            return {"success": False}
        
        # Save results
        output_file = self.save_phrase_templates(phrase_templates)
        
        # Generate statistics
        stats = self.generate_statistics(phrase_templates)
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        # Log results
        logger.info("✅ Phrase mining complete!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Patterns processed: {stats['total_patterns']}")
        logger.info(f"   Phrases extracted: {stats['total_phrases']}")
        logger.info(f"   Runtime: {runtime:.1f}s")
        logger.info(f"   Output: {output_file}")
        
        # Show top patterns
        logger.info(f"🔝 Top patterns by phrase count:")
        for pattern, phrases in stats['top_patterns_by_phrase_count'][:5]:
            logger.info(f"   {pattern}: {len(phrases)} phrases")
        
        return {
            "success": True,
            "output_file": output_file,
            "statistics": stats,
            "runtime_seconds": runtime
        }


def main():
    """Main execution function"""
    try:
        miner = PhraseMiner()
        result = miner.run_mining()
        
        if result["success"]:
            logger.info("🎯 Phrase templates ready for synthetic generation!")
        else:
            logger.error("❌ Phrase mining failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️  Mining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Mining failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 