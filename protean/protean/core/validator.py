#!/usr/bin/env python3
"""
Scenario Validator for Protean Pattern Discovery Engine
Validates infrastructure failure scenarios using pattern matching.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime
from loguru import logger

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None


class PatternExtractor:
    """Extract operations from configuration lines using regex patterns"""
    
    def __init__(self, enable_gpt_fallback: bool = None):
        self.enable_gpt_fallback = enable_gpt_fallback and HAS_OPENAI and bool(os.getenv('OPENAI_API_KEY'))
        self.operation_patterns = self._initialize_patterns()
        self.gpt_cache = {}  # LRU cache for GPT responses
        self.max_cache_size = 1000
        self.openai_client = None
        
        if self.enable_gpt_fallback:
            # Initialize OpenAI client if API key is available
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and HAS_OPENAI:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("🤖 GPT fallback enabled for pattern matching")
            else:
                logger.warning("⚠️  OPENAI_API_KEY not found or openai not installed, disabling GPT fallback")
                self.enable_gpt_fallback = False
        
        logger.info(f"🔍 PatternExtractor initialized with {len(self.operation_patterns)} patterns")
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for matching operations"""
        return {
            'Replicate': [
                r'replicas?\s*[:=]\s*(\d+)',
                r'replication_factor\s*[:=]\s*(\d+)',
                r'backup_count\s*[:=]\s*(\d+)',
                r'mirror_instances\s*[:=]\s*(\d+)',
                r'standby_nodes\s*[:=]\s*(\d+)',
                # Kubernetes spec patterns
                r'spec\.replicas\s*[:=]\s*\d+',
                r'spec\.template\.spec\.replicas\s*[:=]\s*\d+'
            ],
            'Throttle': [
                r'rate_limit\s*[:=]\s*[\d\.]+',
                r'throttle\s*[:=]\s*(true|enabled)',
                r'max_requests_per_second\s*[:=]\s*[\d\.]+',
                r'bandwidth_limit\s*[:=]\s*[\d\.]+',
                r'cpu_throttling\s*[:=]\s*(true|enabled)',
                r'memory_throttling\s*[:=]\s*(true|enabled)',
                # Additional throttling patterns
                r'rateLimit\s*[:=]\s*[\d\.]+',
                r'throttling\.enabled\s*[:=]\s*(true|false)',
                r'request\.rate\s*[:=]\s*[\d\.]+'
            ],
            'Scale': [
                r'(scale|scaling)_policy\s*[:=]',
                r'auto_scaling\s*[:=]\s*(true|enabled)',
                r'min_instances\s*[:=]\s*\d+',
                r'max_instances\s*[:=]\s*\d+',
                r'scaling_threshold\s*[:=]\s*[\d\.]+',
                r'elastic\s*[:=]\s*(true|enabled)',
                # Kubernetes scaling patterns
                r'spec\.minReplicas\s*[:=]\s*\d+',
                r'spec\.maxReplicas\s*[:=]\s*\d+',
                r'autoscaling\s*[:=]\s*(true|enabled)'
            ],
            'Restart': [
                r'restart_policy\s*[:=]',
                r'auto_restart\s*[:=]\s*(true|enabled)',
                r'restart_on_failure\s*[:=]\s*(true|enabled)',
                r'restart_delay\s*[:=]\s*[\d\.]+',
                r'max_restart_attempts\s*[:=]\s*\d+',
                # Kubernetes restart patterns
                r'restartPolicy\s*[:=]\s*(Always|Never|OnFailure)',
                r'spec\.restartPolicy\s*[:=]'
            ],
            'Timeout': [
                r'timeout\s*[:=]\s*[\d\.]+',
                r'connection_timeout\s*[:=]\s*[\d\.]+',
                r'read_timeout\s*[:=]\s*[\d\.]+',
                r'write_timeout\s*[:=]\s*[\d\.]+',
                r'request_timeout\s*[:=]\s*[\d\.]+',
                r'network_timeout\s*[:=]\s*[\d\.]+',
                r'response_timeout\s*[:=]\s*[\d\.]+',
                r'health_check_timeout\s*[:=]\s*[\d\.]+',
                # Java/Spring timeout patterns
                r'connect-timeout-millis\s*[:=]\s*\d+',
                r'socket-timeout\s*[:=]\s*\d+',
                r'redis\.timeout\s*[:=]\s*\d+',
                # Hystrix timeout patterns
                r'hystrix\.command\.default\.execution\.isolation\.thread\.timeoutInMilliseconds\s*[:=]\s*\d+',
                r'hystrix\..*\.timeout\s*[:=]\s*\d+',
                # Spring Boot timeouts
                r'spring\..*\.timeout\s*[:=]\s*[\d\.]+',
                # Kubernetes probe timeouts
                r'probe_timeout\s*[:=]\s*[\d\.]+',
                r'timeoutSeconds\s*[:=]\s*\d+'
            ],
            'CircuitBreaker': [
                r'circuit_breaker\s*[:=]\s*(true|enabled)',
                r'failure_threshold\s*[:=]\s*[\d\.]+',
                r'recovery_timeout\s*[:=]\s*[\d\.]+',
                r'error_rate_threshold\s*[:=]\s*[\d\.]+',
                r'breaker_open_timeout\s*[:=]\s*[\d\.]+',
                # Hystrix circuit breaker patterns
                r'hystrix\.command\.default\.circuitBreaker\.enabled\s*[:=]\s*(true|false)',
                r'hystrix\..*\.circuitBreaker\.',
                r'circuitBreaker\.errorThresholdPercentage\s*[:=]\s*\d+',
                r'circuitBreaker\.requestVolumeThreshold\s*[:=]\s*\d+',
                # Resilience4j patterns
                r'resilience4j\.circuitbreaker\.instances\..*\.failureRateThreshold\s*[:=]\s*\d+',
                r'resilience4j\.circuitbreaker\..*',
                # Additional patterns
                r'circuit_breaker_threshold\s*[:=]\s*\d+',
                r'circuit\.breaker\.',
                r'breaker\..*\.(enabled|threshold)'
            ],
            'Retry': [
                r'retry_policy\s*[:=]',
                r'max_retries\s*[:=]\s*\d+',
                r'retry_delay\s*[:=]\s*[\d\.]+',
                r'exponential_backoff\s*[:=]\s*(true|enabled)',
                r'retry_timeout\s*[:=]\s*[\d\.]+',
                # Spring retry patterns
                r'spring\.retry\.max-attempts\s*[:=]\s*\d+',
                r'spring\.retry\..*',
                # Additional retry patterns
                r'max_retry_attempts\s*[:=]\s*\d+',
                r'backoff_multiplier\s*[:=]\s*[\d\.]+',
                r'retry\..*\.(attempts|delay|backoff)',
                r'retryable\s*[:=]\s*(true|enabled)'
            ],
            'LoadBalance': [
                r'load_balancing\s*[:=]',
                r'load_balancer\s*[:=]',
                r'balancing_algorithm\s*[:=]',
                r'round_robin\s*[:=]\s*(true|enabled)',
                r'sticky_sessions\s*[:=]\s*(true|enabled)',
                r'health_check_interval\s*[:=]\s*[\d\.]+',
                # Load balancer policy patterns
                r'load_balancer_policy\s*[:=]\s*(round_robin|least_conn|ip_hash)',
                r'loadBalancingPolicy\s*[:=]',
                r'upstream_check\s+interval\s*[:=]\s*\d+',
                # Nginx/HAProxy patterns
                r'upstream\s+.*',
                r'server_name_indication\s*[:=]\s*(on|off)',
                # Kubernetes service patterns
                r'sessionAffinity\s*[:=]'
            ],
            'Cache': [
                r'cache\s*[:=]\s*(true|enabled)',
                r'cache_size\s*[:=]\s*[\d\.]+',
                r'cache_ttl\s*[:=]\s*[\d\.]+',
                r'cache_policy\s*[:=]',
                r'redis\s*[:=]',
                r'memcache\s*[:=]',
                # Spring Cache patterns
                r'cache\.expire-after-write\s*[:=]',
                r'spring\.cache\..*',
                # Additional cache patterns
                r'caching\s*[:=]\s*(true|enabled)',
                r'cache\..*\.(size|ttl|policy|timeout)',
                r'redis\..*',
                r'memcached\..*'
            ],
            'Monitor': [
                r'monitoring\s*[:=]\s*(true|enabled)',
                r'health_check\s*[:=]\s*(true|enabled)',
                r'metrics\s*[:=]\s*(true|enabled)',
                r'alerting\s*[:=]\s*(true|enabled)',
                r'log_level\s*[:=]',
                r'trace\s*[:=]\s*(true|enabled)',
                # Spring Boot Actuator patterns
                r'management\.endpoints\.web\.exposure\.include\s*[:=]',
                r'management\..*',
                # Logging patterns
                r'logging\.level\..*\s*[:=]',
                r'logging\..*',
                # Metrics patterns
                r'metrics\.enabled\s*[:=]\s*(true|false)',
                r'micrometer\..*',
                # Kubernetes monitoring
                r'metadata\.name\s*[:=].*monitoring',
                r'spec\.group\s*[:=].*monitoring',
                r'prometheus\.io/.*'
            ],
            'Backup': [
                r'backup\s*[:=]\s*(true|enabled)',
                r'backup_schedule\s*[:=]',
                r'backup_retention\s*[:=]\s*[\d\.]+',
                r'snapshot\s*[:=]\s*(true|enabled)',
                r'backup_location\s*[:=]'
            ],
            'Failover': [
                r'failover\s*[:=]\s*(true|enabled)',
                r'failover_timeout\s*[:=]\s*[\d\.]+',
                r'active_passive\s*[:=]',
                r'standby\s*[:=]\s*(true|enabled)',
                r'primary\s*[:=]',
                r'secondary\s*[:=]'
            ],
            'ResourceLimit': [
                r'memory_limit\s*[:=]\s*[\d\.]+',
                r'cpu_limit\s*[:=]\s*[\d\.]+',
                r'disk_quota\s*[:=]\s*[\d\.]+',
                r'max_connections\s*[:=]\s*\d+',
                r'connection_pool_size\s*[:=]\s*\d+',
                r'resource_quota\s*[:=]',
                # Kubernetes resource patterns
                r'resources\.limits\.memory\s*[:=]',
                r'resources\.limits\.cpu\s*[:=]',
                r'spec\..*\.resources\.',
                # JVM memory patterns
                r'java\..*\.memory',
                r'heap\.size\s*[:=]'
            ],
            'ServiceConfig': [
                r'service_name\s*[:=]\s*[\w\-]+',
                r'service_type\s*[:=]\s*[\w\-]+',
                r'port\s*[:=]\s*\d+',
                r'environment\s*[:=]\s*[\w\-]+',
                r'image\s*[:=]\s*[\w\-\.:\/]+',
                r'namespace\s*[:=]\s*[\w\-]+',
                r'version\s*[:=]\s*[\w\-\.]+',
                r'replica_count\s*[:=]\s*\d+',
                # Enhanced patterns for diverse configs
                r'deployment\.strategy\s*[:=]\s*[\w\-]+',
                r'failure\.context\s*[:=]\s*[\w\-_]+',
                r'scenario\.id\s*[:=]\s*[a-f0-9]+',
                r'pattern\.fingerprint\s*[:=]\s*[\w\-_]+',
                # Kubernetes service patterns
                r'kind\s*[:=]\s*(Service|ServiceAccount|Deployment)',
                r'metadata\.labels\.service\s*[:=]',
                r'metadata\.labels\.app\s*[:=]',
                r'spec\.template\.spec\.serviceAccountName\s*[:=]',
                r'spec\.selector\..*',
                r'spec\.type\s*[:=]\s*(ClusterIP|NodePort|LoadBalancer)',
                # Service mesh patterns
                r'service\..*',
                r'sidecar\..*'
            ],
            'ErrorHandling': [
                r'error_rate_threshold\s*[:=]\s*[\d\.]+',
                r'failure_threshold\s*[:=]\s*[\d\.]+',
                r'error_budget\s*[:=]\s*[\d\.]+',
                r'max_error_rate\s*[:=]\s*[\d\.]+',
                r'error_policy\s*[:=]',
                r'failure_mode\s*[:=]',
                # Enhanced patterns for variable error rates
                r'error_rate_threshold\s*[:=]\s*0\.\d{3}',  # Match 0.xxx format
                r'error\.threshold\s*[:=]\s*[\d\.]+',
                r'failure\.rate\s*[:=]\s*[\d\.]+',
                r'error\.budget\.percent\s*[:=]\s*[\d\.]+',
                r'threshold\.error_rate\s*[:=]\s*[\d\.]+',
                r'slo\.error_rate\s*[:=]\s*[\d\.]+'
            ],
            'NetworkConfig': [
                r'network_timeout\s*[:=]\s*[\d\.]+',
                r'connection_timeout\s*[:=]\s*[\d\.]+',
                r'keep_alive\s*[:=]\s*(true|false|enabled|disabled)',
                r'max_idle_connections\s*[:=]\s*\d+',
                r'network_policy\s*[:=]',
                r'proxy_config\s*[:=]'
            ],
            'SecurityPolicy': [
                r'security_policy\s*[:=]',
                r'encryption\s*[:=]\s*(true|enabled)',
                r'auth_required\s*[:=]\s*(true|enabled)',
                r'ssl_enabled\s*[:=]\s*(true|enabled)',
                r'certificate_validation\s*[:=]',
                r'oauth_config\s*[:=]',
                # Dot notation security patterns
                r'auth\.enabled\s*[:=]\s*(true|false)',
                r'oauth2\.enabled\s*[:=]\s*(true|false)',
                r'encryption\.enabled\s*[:=]\s*(true|false)',
                r'security\.require-ssl\s*[:=]\s*(true|false)',
                r'ssl\..*',
                r'tls\..*',
                # Spring Security patterns
                r'spring\.security\..*',
                r'security\..*\.(enabled|required)'
            ],
            'ConfigReload': [
                r'config_reload\s*[:=]\s*(true|enabled)',
                r'hot_reload\s*[:=]\s*(true|enabled)',
                r'config_validation\s*[:=]',
                r'reload_signal\s*[:=]',
                r'watch_config\s*[:=]\s*(true|enabled)'
            ],
            'QueueManagement': [
                r'queue_size\s*[:=]\s*\d+',
                r'queue_timeout\s*[:=]\s*[\d\.]+',
                r'dead_letter_queue\s*[:=]\s*(true|enabled)',
                r'message_retry_count\s*[:=]\s*\d+',
                r'queue_overflow\s*[:=]',
                r'consumer_lag\s*[:=]'
            ],
            'DataPersistence': [
                r'message_persistence\s*[:=]\s*(true|enabled)',
                r'ack_timeout\s*[:=]\s*[\d\.]+',
                r'max_message_size\s*[:=]',
                r'persistence_layer\s*[:=]',
                r'data_retention\s*[:=]'
            ],
            'HealthCheck': [
                r'health_check_path\s*[:=]',
                r'backend_health_check\s*[:=]\s*(true|enabled)',
                r'probe_interval\s*[:=]\s*[\d\.]+',
                r'readiness_probe\s*[:=]',
                r'liveness_probe\s*[:=]',
                # Kubernetes health check patterns
                r'livenessProbe\.',
                r'readinessProbe\.',
                r'healthcheck\.',
                # HTTP health checks
                r'health\.check\..*'
            ],
            'Bulkhead': [
                r'bulkhead_isolation\s*[:=]\s*(true|enabled)',
                r'isolation_level\s*[:=]',
                r'thread_pool_isolation\s*[:=]',
                r'resource_isolation\s*[:=]'
            ],
            'FallbackService': [
                r'fallback_service\s*[:=]\s*(true|enabled)',
                r'fallback_timeout\s*[:=]\s*[\d\.]+',
                r'default_response\s*[:=]',
                r'fallback_strategy\s*[:=]'
            ],
            'DeploymentStrategy': [
                r'deployment_strategy\s*[:=]',
                r'rolling_update_max_surge\s*[:=]',
                r'rolling_update_max_unavailable\s*[:=]',
                r'blue_green_deployment\s*[:=]',
                r'canary_deployment\s*[:=]'
            ]
        }
    
    def _match_line_to_operation(self, line: str) -> Optional[str]:
        """Match a configuration line to an operation using regex patterns"""
        line = line.strip().lower()
        
        # Try regex patterns first
        for operation, patterns in self.operation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    logger.debug(f"✅ Matched '{line}' to '{operation}' via regex")
                    return operation
        
        # Fall back to GPT if enabled
        if self.enable_gpt_fallback:
            return self._gpt_fallback_match(line)
        
        return None
    
    def _gpt_fallback_match(self, line: str) -> Optional[str]:
        """Use GPT to match configuration line to operation"""
        if not self.openai_client:
            return None
            
        # Check cache first
        if line in self.gpt_cache:
            logger.debug(f"🎯 GPT cache hit for '{line}'")
            return self.gpt_cache[line]
        
        try:
            # Prepare prompt
            operation_list = list(self.operation_patterns.keys())
            prompt = f"""Return single token from this list for the config line:
{', '.join(operation_list)}

Config line: {line}

Return only the operation token, nothing else."""
            
            # Call OpenAI API with new client
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate result
            if result in operation_list:
                # Cache the result
                self._cache_gpt_result(line, result)
                logger.debug(f"🤖 GPT matched '{line}' to '{result}'")
                return result
            else:
                logger.warning(f"⚠️  GPT returned invalid operation: '{result}'")
                return None
                
        except Exception as e:
            logger.error(f"❌ GPT fallback failed for '{line}': {e}")
            return None
    
    def _cache_gpt_result(self, line: str, result: str) -> None:
        """Cache GPT result with LRU eviction"""
        if len(self.gpt_cache) >= self.max_cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.gpt_cache))
            del self.gpt_cache[oldest_key]
        
        self.gpt_cache[line] = result


class ScenarioValidator:
    """Validate infrastructure failure scenarios"""
    
    def __init__(self, data_dir: str = "data/smoke", output_dir: str = "data/diagnostics", max_scenarios: int = 50):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_scenarios = max_scenarios
        self.extractor = PatternExtractor()
        
        logger.info(f"🔍 ScenarioValidator initialized")
        logger.info(f"   Data dir: {self.data_dir}")
        logger.info(f"   Output dir: {self.output_dir}")
        logger.info(f"   Max scenarios: {self.max_scenarios}")
    
    def validate_scenarios(self) -> Dict[str, Any]:
        """Validate scenarios and return results"""
        start_time = datetime.now()
        
        # Load configuration lines
        config_lines = self._load_config_lines()
        if not config_lines:
            logger.error("❌ No configuration lines found for validation")
            return {"success": False, "error": "No config lines found"}
        
        # Validate lines against patterns
        results = self._validate_lines(config_lines)
        
        # Generate diagnostics
        self._generate_diagnostics(results, config_lines)
        
        # Calculate final metrics
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        total_lines = len(config_lines)
        matched_lines = results['matched_count']
        accuracy = (matched_lines / total_lines * 100) if total_lines > 0 else 0
        
        final_results = {
            'success': True,
            'total_scenarios': results['scenarios_processed'],
            'total_lines': total_lines,
            'matched_lines': matched_lines,
            'unmatched_lines': total_lines - matched_lines,
            'accuracy': accuracy,
            'runtime_seconds': runtime,
            'operation_distribution': results['operation_counts'],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_file = self.output_dir / "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"📊 Validation complete: {accuracy:.1f}% accuracy ({matched_lines}/{total_lines} lines)")
        return final_results
    
    def _load_config_lines(self) -> List[str]:
        """Load configuration lines from scenarios"""
        config_lines = []
        scenarios_processed = 0
        
        # Look for config_lines.txt first
        config_file = self.data_dir / "scenarios" / "config_lines.txt"
        if config_file.exists():
            with open(config_file, 'r') as f:
                lines = f.readlines()
                # Skip comments and empty lines
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        config_lines.append(line)
            logger.info(f"📄 Loaded {len(config_lines)} config lines from {config_file}")
            return config_lines
        
        # Fall back to extracting from scenario files
        scenarios_dir = self.data_dir / "scenarios"
        if not scenarios_dir.exists():
            logger.error(f"❌ Scenarios directory not found: {scenarios_dir}")
            return []
        
        scenario_files = list(scenarios_dir.glob("scenario_*.json"))
        if not scenario_files:
            logger.error(f"❌ No scenario files found in {scenarios_dir}")
            return []
        
        # Process scenario files
        for scenario_file in scenario_files[:self.max_scenarios]:
            try:
                with open(scenario_file, 'r') as f:
                    scenario = json.load(f)
                
                # Extract config lines from injection specs
                for event in scenario.get('events', []):
                    injection_spec = event.get('injection_spec')
                    if injection_spec:
                        # Create config lines from injection spec
                        config_lines.extend(self._extract_config_from_injection(injection_spec))
                
                scenarios_processed += 1
                
            except Exception as e:
                logger.warning(f"⚠️  Error processing {scenario_file}: {e}")
        
        logger.info(f"📄 Extracted {len(config_lines)} config lines from {scenarios_processed} scenarios")
        return config_lines
    
    def _extract_config_from_injection(self, injection_spec: Dict[str, Any]) -> List[str]:
        """Extract configuration lines from injection specification"""
        config_lines = []
        
        spec_type = injection_spec.get('type', '')
        target = injection_spec.get('target', 'unknown')
        severity = injection_spec.get('severity', 'medium')
        duration = injection_spec.get('duration', 30)
        metrics = injection_spec.get('metrics', {})
        
        # Generate config lines based on injection type
        if spec_type == 'timeout':
            config_lines.append(f"timeout: {duration}s")
            config_lines.append(f"connection_timeout: {duration + 10}s")
            config_lines.append("retry_policy: exponential_backoff")
            config_lines.append("max_retries: 3")
            
        elif spec_type == 'failure':
            error_rate = metrics.get('error_rate', 0.1)
            config_lines.append(f"error_rate_threshold: {error_rate}")
            config_lines.append("circuit_breaker: enabled")
            config_lines.append("failure_threshold: 5")
            config_lines.append("recovery_timeout: 60s")
            
        elif spec_type == 'degradation':
            config_lines.append("throttle: enabled")
            config_lines.append("rate_limit: 100")
            config_lines.append("cpu_throttling: enabled")
            config_lines.append("memory_throttling: enabled")
            
        elif spec_type == 'pod_restart':
            config_lines.append("restart_policy: always")
            config_lines.append("auto_restart: enabled")
            config_lines.append("max_restart_attempts: 3")
            config_lines.append("restart_delay: 5s")
            
        elif spec_type == 'scaling':
            config_lines.append("auto_scaling: enabled")
            config_lines.append("min_instances: 2")
            config_lines.append("max_instances: 10")
            config_lines.append("scaling_threshold: 0.7")
            
        elif spec_type == 'deployment':
            config_lines.append("deployment_strategy: rolling")
            config_lines.append("rolling_update_max_surge: 25%")
            config_lines.append("rolling_update_max_unavailable: 25%")
            
        elif spec_type == 'config_change':
            config_lines.append("config_reload: enabled")
            config_lines.append("hot_reload: true")
            config_lines.append("config_validation: strict")
            
        elif spec_type == 'connection_loss':
            config_lines.append("connection_pool_size: 20")
            config_lines.append("connection_timeout: 30s")
            config_lines.append("keep_alive: enabled")
            config_lines.append("max_idle_connections: 10")
            
        elif spec_type == 'queue_overflow':
            config_lines.append("queue_size: 1000")
            config_lines.append("queue_timeout: 30s")
            config_lines.append("dead_letter_queue: enabled")
            config_lines.append("message_retry_count: 3")
            
        elif spec_type == 'traffic_spike':
            config_lines.append("load_balancing: round_robin")
            config_lines.append("rate_limit: 1000")
            config_lines.append("health_check_interval: 10s")
            config_lines.append("backend_health_check: enabled")
            
        elif spec_type == 'backend_failure':
            config_lines.append("failover: enabled")
            config_lines.append("failover_timeout: 30s")
            config_lines.append("backup_instances: 2")
            config_lines.append("health_check_path: /health")
            
        elif spec_type == 'message_loss':
            config_lines.append("message_persistence: enabled")
            config_lines.append("ack_timeout: 30s")
            config_lines.append("retry_policy: exponential")
            config_lines.append("max_message_size: 1MB")
            
        elif spec_type == 'initial_failure' or spec_type == 'propagation' or spec_type == 'cascade':
            config_lines.append("circuit_breaker: enabled")
            config_lines.append("bulkhead_isolation: enabled")
            config_lines.append("timeout: 30s")
            config_lines.append("fallback_service: enabled")
            
        else:
            # Default patterns for unknown types
            config_lines.append(f"timeout: {duration}s")
            config_lines.append("monitoring: enabled")
            config_lines.append("alerting: enabled")
        
        # Add more diverse target-specific config based on severity
        config_lines.append(f"service_name: {target}")
        
        if severity == 'high':
            config_lines.append("replicas: 5")
            config_lines.append("memory_limit: 1GB")
            config_lines.append("cpu_limit: 0.8")
            config_lines.append("health_check_interval: 5s")
        elif severity == 'medium':
            config_lines.append("replicas: 3")
            config_lines.append("memory_limit: 512MB")
            config_lines.append("cpu_limit: 0.5")
            config_lines.append("health_check_interval: 15s")
        else:
            config_lines.append("replicas: 2")
            config_lines.append("memory_limit: 256MB")
            config_lines.append("cpu_limit: 0.3")
            config_lines.append("health_check_interval: 30s")
        
        # Add caching and monitoring patterns
        config_lines.append("cache: enabled")
        config_lines.append("cache_ttl: 300s")
        config_lines.append("metrics: enabled")
        config_lines.append("log_level: info")
        
        # Add backup and storage patterns
        config_lines.append("backup: enabled")
        config_lines.append("backup_schedule: daily")
        config_lines.append("disk_quota: 10GB")
        
        # Add novel patterns based on injection type and severity
        if spec_type == 'failure' or severity == 'high':
            # Security and isolation patterns for critical failures
            config_lines.append("encryption: enabled")
            config_lines.append("bulkhead_isolation: enabled")
            config_lines.append("security_policy: strict")
            config_lines.append("auth_required: enabled")
            
        if spec_type in ['config_change', 'deployment']:
            # Config and deployment patterns
            config_lines.append("config_reload: enabled")
            config_lines.append("hot_reload: true")
            config_lines.append("config_validation: strict")
            config_lines.append("deployment_strategy: rolling")
            config_lines.append("rolling_update_max_surge: 25%")
            
        if spec_type in ['queue_overflow', 'message_loss']:
            # Queue management patterns
            config_lines.append("queue_size: 1000")
            config_lines.append("dead_letter_queue: enabled")
            config_lines.append("message_persistence: enabled")
            config_lines.append("ack_timeout: 30s")
            config_lines.append("max_message_size: 1MB")
            
        if spec_type in ['backend_failure', 'connection_loss']:
            # Health check and failover patterns
            config_lines.append("health_check_path: /health")
            config_lines.append("backend_health_check: enabled")
            config_lines.append("fallback_service: enabled")
            config_lines.append("readiness_probe: /ready")
            config_lines.append("liveness_probe: /health")
            
        # Add resource and network patterns based on target
        if 'network' in target or 'proxy' in target or 'load-balancer' in target:
            config_lines.append("keep_alive: enabled")
            config_lines.append("ssl_enabled: enabled")
            config_lines.append("proxy_config: upstream")
            config_lines.append("network_policy: allow_selected")
            
        # Add scaling and management patterns
        config_lines.append("auto_scaling: enabled")
        config_lines.append("min_instances: 2")
        config_lines.append("max_instances: 10")
        config_lines.append("thread_pool_isolation: enabled")
        
        return config_lines
    
    def _validate_lines(self, config_lines: List[str]) -> Dict[str, Any]:
        """Validate configuration lines against operation patterns"""
        matched_lines = []
        unmatched_lines = []
        operation_counts = defaultdict(int)
        
        for line in config_lines:
            operation = self.extractor._match_line_to_operation(line)
            if operation:
                matched_lines.append((line, operation))
                operation_counts[operation] += 1
            else:
                unmatched_lines.append(line)
        
        # Count actual scenarios processed from scenario files
        scenarios_dir = self.data_dir / "scenarios"
        scenario_count = len(list(scenarios_dir.glob("scenario_*.json"))) if scenarios_dir.exists() else 0
        
        return {
            'matched_lines': matched_lines,
            'unmatched_lines': unmatched_lines,
            'matched_count': len(matched_lines),
            'unmatched_count': len(unmatched_lines),
            'operation_counts': dict(operation_counts),
            'scenarios_processed': scenario_count
        }
    
    def _generate_diagnostics(self, results: Dict[str, Any], all_lines: List[str]) -> None:
        """Generate diagnostic files"""
        timestamp = datetime.now().isoformat()
        
        # Write unmatched lines
        unmatched_file = self.output_dir / "unmatched_lines.log"
        with open(unmatched_file, 'w') as f:
            f.write(f"# Unmatched configuration lines - {timestamp}\n")
            f.write(f"# Total unmatched: {results['unmatched_count']}\n\n")
            
            for line in results['unmatched_lines']:
                f.write(f"{line}\n")
        
        # Write matched lines
        matched_file = self.output_dir / "matched_lines.log"
        with open(matched_file, 'w') as f:
            f.write(f"# Matched configuration lines - {timestamp}\n")
            f.write(f"# Total matched: {results['matched_count']}\n\n")
            
            for line, operation in results['matched_lines']:
                f.write(f"{operation:<15} | {line}\n")
        
        # Write operation distribution
        stats_file = self.output_dir / "operation_stats.json"
        with open(stats_file, 'w') as f:
            stats = {
                'timestamp': timestamp,
                'total_lines': len(all_lines),
                'matched_lines': results['matched_count'],
                'unmatched_lines': results['unmatched_count'],
                'accuracy_percent': (results['matched_count'] / len(all_lines) * 100) if all_lines else 0,
                'operation_distribution': results['operation_counts'],
                'top_operations': sorted(results['operation_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
            }
            json.dump(stats, f, indent=2)
        
        logger.info(f"📋 Diagnostics written to {self.output_dir}")
        logger.info(f"   Unmatched lines: {unmatched_file}")
        logger.info(f"   Matched lines: {matched_file}")
        logger.info(f"   Statistics: {stats_file}")

    def validate_external_dataset(self, external_config_path: str, ground_truth_path: str) -> Dict[str, Any]:
        """
        Validate against external dataset to avoid circular validation
        This provides real accuracy metrics independent of our training data
        """
        start_time = datetime.now()
        
        # Load external config lines
        external_lines = []
        try:
            with open(external_config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        external_lines.append(line)
        except FileNotFoundError:
            logger.error(f"❌ External config file not found: {external_config_path}")
            return {"success": False, "error": "External config file not found"}
        
        if not external_lines:
            logger.error("❌ No external config lines found")
            return {"success": False, "error": "No external lines found"}
        
        # Load ground truth labels
        ground_truth = {}
        try:
            with open(ground_truth_path, 'r') as f:
                truth_data = json.load(f)
                for item in truth_data:
                    ground_truth[item['config_line']] = item['expected_pattern']
        except FileNotFoundError:
            logger.warning(f"⚠️  Ground truth file not found: {ground_truth_path}")
            ground_truth = {}
        
        # Validate lines against patterns
        results = self._validate_external_lines(external_lines, ground_truth)
        
        # Generate external diagnostics
        self._generate_external_diagnostics(results, external_lines, ground_truth)
        
        # Calculate final metrics
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        total_lines = len(external_lines)
        matched_lines = results['matched_count']
        accuracy = (matched_lines / total_lines * 100) if total_lines > 0 else 0
        
        # Calculate ground truth accuracy if available
        ground_truth_accuracy = 0
        correct_predictions = 0
        if ground_truth:
            for line, predicted_pattern in results['matched_lines']:
                if line in ground_truth:
                    if predicted_pattern == ground_truth[line]:
                        correct_predictions += 1
            
            ground_truth_total = len([line for line in external_lines if line in ground_truth])
            ground_truth_accuracy = (correct_predictions / ground_truth_total * 100) if ground_truth_total > 0 else 0
        
        final_results = {
            'success': True,
            'data_source': 'external',
            'total_lines': total_lines,
            'matched_lines': matched_lines,
            'unmatched_lines': total_lines - matched_lines,
            'pattern_coverage_accuracy': accuracy,  # % of lines that matched any pattern
            'ground_truth_accuracy': ground_truth_accuracy,  # % of correct pattern predictions
            'correct_predictions': correct_predictions,
            'ground_truth_total': len(ground_truth) if ground_truth else 0,
            'runtime_seconds': runtime,
            'operation_distribution': results['operation_counts'],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Save external validation results
        results_file = self.output_dir / "external_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"📊 External validation complete:")
        logger.info(f"   Pattern coverage: {accuracy:.1f}% ({matched_lines}/{total_lines} lines)")
        if ground_truth:
            logger.info(f"   Ground truth accuracy: {ground_truth_accuracy:.1f}% ({correct_predictions}/{len(ground_truth)} labeled lines)")
        
        return final_results
    
    def _validate_external_lines(self, external_lines: List[str], ground_truth: Dict[str, str]) -> Dict[str, Any]:
        """Validate external configuration lines against operation patterns"""
        matched_lines = []
        unmatched_lines = []
        operation_counts = defaultdict(int)
        
        for line in external_lines:
            operation = self.extractor._match_line_to_operation(line)
            if operation:
                matched_lines.append((line, operation))
                operation_counts[operation] += 1
                logger.debug(f"External match: '{line}' → '{operation}'")
            else:
                unmatched_lines.append(line)
                logger.debug(f"External unmatch: '{line}'")
        
        return {
            'matched_lines': matched_lines,
            'unmatched_lines': unmatched_lines,
            'matched_count': len(matched_lines),
            'unmatched_count': len(unmatched_lines),
            'operation_counts': dict(operation_counts)
        }
    
    def _generate_external_diagnostics(self, results: Dict[str, Any], all_lines: List[str], ground_truth: Dict[str, str]) -> None:
        """Generate diagnostic files for external validation"""
        timestamp = datetime.now().isoformat()
        
        # Write external unmatched lines
        unmatched_file = self.output_dir / "external_unmatched_lines.log"
        with open(unmatched_file, 'w') as f:
            f.write(f"# External unmatched configuration lines - {timestamp}\n")
            f.write(f"# Total unmatched: {results['unmatched_count']}\n\n")
            
            for line in results['unmatched_lines']:
                expected = ground_truth.get(line, 'Unknown')
                f.write(f"{line} [Expected: {expected}]\n")
        
        # Write external matched lines with accuracy info
        matched_file = self.output_dir / "external_matched_lines.log"
        with open(matched_file, 'w') as f:
            f.write(f"# External matched configuration lines - {timestamp}\n")
            f.write(f"# Total matched: {results['matched_count']}\n\n")
            
            for line, operation in results['matched_lines']:
                expected = ground_truth.get(line, 'Unknown')
                correctness = "✓" if expected == operation else "✗" if expected != 'Unknown' else "?"
                f.write(f"{operation:<15} | {line} [Expected: {expected}] {correctness}\n")
        
        # Write external operation statistics
        external_stats_file = self.output_dir / "external_operation_stats.json"
        with open(external_stats_file, 'w') as f:
            stats = {
                'timestamp': timestamp,
                'data_source': 'external_real_configs',
                'total_lines': len(all_lines),
                'matched_lines': results['matched_count'],
                'unmatched_lines': results['unmatched_count'],
                'pattern_coverage_percent': (results['matched_count'] / len(all_lines) * 100) if all_lines else 0,
                'operation_distribution': results['operation_counts'],
                'top_operations': sorted(results['operation_counts'].items(), key=lambda x: x[1], reverse=True)[:10],
                'ground_truth_samples': len(ground_truth)
            }
            json.dump(stats, f, indent=2)
        
        logger.info(f"📋 External diagnostics written to {self.output_dir}")
        logger.info(f"   External unmatched: {unmatched_file}")
        logger.info(f"   External matched: {matched_file}")
        logger.info(f"   External stats: {external_stats_file}") 