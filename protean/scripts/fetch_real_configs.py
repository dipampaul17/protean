#!/usr/bin/env python3
"""
Fetch Real Infrastructure Configurations for External Validation
Solves the circular validation problem by getting independent data.
"""

import requests
import yaml
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import time
import re
from loguru import logger

class RealConfigFetcher:
    """Fetch real infrastructure configurations for external validation"""
    
    def __init__(self, output_dir: str = "data/external"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Real Kubernetes configurations from popular projects
        self.k8s_configs = [
            # Popular open source projects with real K8s configs
            "https://raw.githubusercontent.com/kubernetes/examples/master/guestbook/all-in-one/guestbook-all-in-one.yaml",
            "https://raw.githubusercontent.com/kubernetes/website/main/content/en/examples/controllers/nginx-deployment.yaml",
            "https://raw.githubusercontent.com/kubernetes/website/main/content/en/examples/service/networking/ingress.yaml",
            "https://raw.githubusercontent.com/istio/istio/master/samples/bookinfo/platform/kube/bookinfo.yaml",
            "https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/example/prometheus-operator-crd/monitoring.coreos.com_prometheuses.yaml",
            "https://raw.githubusercontent.com/grafana/grafana/main/devenv/docker/ha-test/docker-compose.yaml",
        ]
        
        # Common infrastructure patterns from documentation
        self.pattern_examples = {
            "circuit_breaker": [
                "hystrix.command.default.circuitBreaker.enabled=true",
                "circuitBreaker.errorThresholdPercentage=50",
                "resilience4j.circuitbreaker.instances.backend.failureRateThreshold=60",
                "circuit_breaker_threshold: 5",
                "failure_threshold: 3"
            ],
            "timeout": [
                "connect-timeout-millis: 5000",
                "hystrix.command.default.execution.isolation.thread.timeoutInMilliseconds=10000",
                "timeout: 30s",
                "request_timeout: 5000ms",
                "connection_timeout: 10s"
            ],
            "retry": [
                "spring.retry.max-attempts=3",
                "max_retries: 5",
                "retry_policy: exponential_backoff",
                "backoff_multiplier: 2.0",
                "retry_delay: 1000ms"
            ],
            "load_balancing": [
                "load_balancer_policy: round_robin",
                "sticky_sessions: true",
                "health_check_interval: 30s",
                "upstream_check interval=3000",
                "server_name_indication: on"
            ],
            "caching": [
                "redis.timeout=2000",
                "cache.expire-after-write=PT10M",
                "cache_ttl: 300",
                "cache_size: 1000",
                "cache_policy: lru"
            ],
            "security": [
                "security.require-ssl=true",
                "auth.enabled=true",
                "oauth2.enabled=true",
                "encryption.enabled=true",
                "certificate_validation: strict"
            ],
            "monitoring": [
                "management.endpoints.web.exposure.include=health,info,metrics",
                "logging.level.com.example=DEBUG",
                "metrics.enabled=true",
                "health_check_path: /health",
                "probe_timeout: 5s"
            ]
        }
    
    def fetch_k8s_configs(self) -> List[str]:
        """Fetch real Kubernetes configurations"""
        config_lines = []
        
        for url in self.k8s_configs:
            try:
                logger.info(f"Fetching {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Parse YAML and extract configuration lines
                try:
                    docs = yaml.safe_load_all(response.text)
                    for doc in docs:
                        if doc:
                            lines = self._extract_config_lines_from_yaml(doc)
                            config_lines.extend(lines)
                except yaml.YAMLError:
                    # If not YAML, try to extract key-value pairs
                    lines = self._extract_config_lines_from_text(response.text)
                    config_lines.extend(lines)
                    
                time.sleep(0.5)  # Be respectful
                
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
        
        return config_lines
    
    def _extract_config_lines_from_yaml(self, doc: Dict[str, Any]) -> List[str]:
        """Extract configuration lines from YAML document"""
        config_lines = []
        
        def extract_from_dict(d: Dict[str, Any], prefix: str = ""):
            for key, value in d.items():
                if isinstance(value, dict):
                    extract_from_dict(value, f"{prefix}{key}.")
                elif isinstance(value, (str, int, float, bool)):
                    # Extract meaningful configuration patterns
                    line = f"{prefix}{key}: {value}"
                    if self._is_meaningful_config(line):
                        config_lines.append(line)
        
        if isinstance(doc, dict):
            extract_from_dict(doc)
        
        return config_lines
    
    def _extract_config_lines_from_text(self, text: str) -> List[str]:
        """Extract configuration lines from plain text"""
        config_lines = []
        
        # Pattern for key-value configurations
        patterns = [
            r'(\w+(?:\.\w+)*)\s*[:=]\s*([^\n\r]+)',  # key: value or key=value
            r'(\w+)\s*=\s*([^\n\r]+)',  # key=value
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for key, value in matches:
                line = f"{key}: {value.strip()}"
                if self._is_meaningful_config(line):
                    config_lines.append(line)
        
        return config_lines
    
    def _is_meaningful_config(self, line: str) -> bool:
        """Check if a configuration line is meaningful for pattern matching"""
        line_lower = line.lower()
        
        # Infrastructure-related keywords
        keywords = [
            'timeout', 'retry', 'replicas', 'memory', 'cpu', 'port', 'health',
            'circuit', 'cache', 'load', 'balance', 'monitor', 'log', 'auth',
            'security', 'ssl', 'tls', 'backup', 'scale', 'threshold', 'limit',
            'interval', 'policy', 'strategy', 'enabled', 'disabled', 'service',
            'container', 'image', 'volume', 'storage', 'network', 'proxy'
        ]
        
        return any(keyword in line_lower for keyword in keywords)
    
    def create_external_dataset(self) -> str:
        """Create comprehensive external validation dataset"""
        all_config_lines = []
        
        # 1. Fetch real K8s configurations
        logger.info("📥 Fetching real Kubernetes configurations...")
        k8s_lines = self.fetch_k8s_configs()
        all_config_lines.extend(k8s_lines)
        logger.info(f"✅ Extracted {len(k8s_lines)} lines from K8s configs")
        
        # 2. Add documented infrastructure patterns
        logger.info("📚 Adding documented infrastructure patterns...")
        for category, patterns in self.pattern_examples.items():
            all_config_lines.extend(patterns)
        logger.info(f"✅ Added {sum(len(p) for p in self.pattern_examples.values())} documented patterns")
        
        # 3. Remove duplicates and sort
        unique_lines = sorted(set(all_config_lines))
        
        # 4. Write to file
        output_file = self.output_dir / "external_config_lines.txt"
        with open(output_file, 'w') as f:
            f.write(f"# External Infrastructure Configuration Lines for Validation\n")
            f.write(f"# Source: Real K8s configs + documented patterns\n")
            f.write(f"# Total lines: {len(unique_lines)}\n")
            f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for line in unique_lines:
                f.write(f"{line}\n")
        
        logger.info(f"💾 Saved {len(unique_lines)} external config lines to {output_file}")
        return str(output_file)
    
    def create_ground_truth_labels(self, config_file: str) -> str:
        """Create ground truth labels for external dataset"""
        # This would normally be done by domain experts
        # For now, we'll create a basic mapping based on keywords
        
        label_mapping = {
            'timeout': 'Timeout',
            'retry': 'Retry', 
            'circuit': 'CircuitBreaker',
            'replicas': 'Replicate',
            'cache': 'Cache',
            'health': 'Monitor',
            'load': 'LoadBalance',
            'auth': 'SecurityPolicy',
            'memory': 'ResourceLimit',
            'cpu': 'ResourceLimit',
            'service': 'ServiceConfig',
            'backup': 'Backup',
            'scale': 'Scale',
            'monitor': 'Monitor',
            'log': 'Monitor'
        }
        
        ground_truth = []
        with open(config_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Find best matching label
                    best_label = 'Unknown'
                    for keyword, label in label_mapping.items():
                        if keyword in line.lower():
                            best_label = label
                            break
                    
                    ground_truth.append({
                        'line_number': line_num,
                        'config_line': line,
                        'expected_pattern': best_label
                    })
        
        # Save ground truth
        truth_file = self.output_dir / "ground_truth_labels.json"
        with open(truth_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        logger.info(f"📋 Created ground truth labels: {truth_file}")
        return str(truth_file)

def main():
    """Create external validation dataset"""
    fetcher = RealConfigFetcher()
    
    # Create external dataset
    config_file = fetcher.create_external_dataset()
    
    # Create ground truth labels  
    truth_file = fetcher.create_ground_truth_labels(config_file)
    
    print(f"✅ External validation dataset ready:")
    print(f"   Config lines: {config_file}")
    print(f"   Ground truth: {truth_file}")

if __name__ == "__main__":
    main() 