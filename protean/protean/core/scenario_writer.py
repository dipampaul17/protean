#!/usr/bin/env python3
"""
Scenario Writer for Protean Pattern Discovery Engine
Writes generated scenarios to files for validation and testing.
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from loguru import logger

from protean.synthesis.scenarios.generator import Scenario, ScenarioEvent, InjectionSpec


class ScenarioWriter:
    """Write scenarios to files in various formats"""
    
    def __init__(self, output_dir: str = "data/smoke/scenarios"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 ScenarioWriter initialized with output dir: {self.output_dir}")
    
    def write_scenarios(self, scenarios: List[Scenario]) -> List[str]:
        """Write all scenarios to files"""
        output_files = []
        
        # Write individual scenario files
        for i, scenario in enumerate(scenarios):
            # Write as JSON for validation
            json_file = self.output_dir / f"scenario_{i+1:03d}_{scenario.category.lower()}.json"
            self._write_scenario_json(scenario, json_file)
            output_files.append(str(json_file))
            
            # Write as YAML for human readability
            yaml_file = self.output_dir / f"scenario_{i+1:03d}_{scenario.category.lower()}.yaml"
            self._write_scenario_yaml(scenario, yaml_file)
            output_files.append(str(yaml_file))
        
        # Write summary file
        summary_file = self.output_dir / "scenarios_summary.json"
        self._write_summary(scenarios, summary_file)
        output_files.append(str(summary_file))
        
        # Write config lines for pattern matching
        config_file = self.output_dir / "config_lines.txt"
        self._write_config_lines(scenarios, config_file)
        output_files.append(str(config_file))
        
        logger.info(f"✅ Written {len(scenarios)} scenarios to {len(output_files)} files")
        return output_files
    
    def _write_scenario_json(self, scenario: Scenario, file_path: Path) -> None:
        """Write scenario as JSON"""
        data = self._scenario_to_dict(scenario)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _write_scenario_yaml(self, scenario: Scenario, file_path: Path) -> None:
        """Write scenario as YAML"""
        data = self._scenario_to_dict(scenario)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
    
    def _scenario_to_dict(self, scenario: Scenario) -> Dict[str, Any]:
        """Convert scenario to dictionary"""
        return {
            'category': scenario.category,
            'name': scenario.name,
            'description': scenario.description,
            'duration': scenario.duration,
            'metadata': scenario.metadata,
            'events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'description': event.description,
                    'injection_spec': {
                        'type': event.injection_spec.type,
                        'target': event.injection_spec.target,
                        'severity': event.injection_spec.severity,
                        'duration': event.injection_spec.duration,
                        'log_snippet': event.injection_spec.log_snippet,
                        'metrics': event.injection_spec.metrics
                    } if event.injection_spec else None
                }
                for event in scenario.events
            ]
        }
    
    def _write_summary(self, scenarios: List[Scenario], file_path: Path) -> None:
        """Write summary of all scenarios"""
        # Calculate statistics
        category_counts = {}
        total_events = 0
        total_injection_specs = 0
        
        for scenario in scenarios:
            category_counts[scenario.category] = category_counts.get(scenario.category, 0) + 1
            total_events += len(scenario.events)
            total_injection_specs += sum(1 for event in scenario.events if event.injection_spec)
        
        summary = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_scenarios': len(scenarios),
                'total_events': total_events,
                'total_injection_specs': total_injection_specs,
                'categories': len(category_counts)
            },
            'category_distribution': category_counts,
            'scenarios': [
                {
                    'id': i + 1,
                    'category': scenario.category,
                    'name': scenario.name,
                    'events_count': len(scenario.events),
                    'injection_specs_count': sum(1 for event in scenario.events if event.injection_spec),
                    'duration': scenario.duration
                }
                for i, scenario in enumerate(scenarios)
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    
    def _write_config_lines(self, scenarios: List[Scenario], file_path: Path) -> None:
        """Write configuration lines for pattern matching validation"""
        config_lines = []
        
        for scenario in scenarios:
            for event in scenario.events:
                if event.injection_spec:
                    # Create diverse config lines based on the injection spec
                    spec_type = event.injection_spec.type
                    target = event.injection_spec.target
                    severity = event.injection_spec.severity
                    duration = event.injection_spec.duration
                    metrics = event.injection_spec.metrics
                    
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
                        
                    elif spec_type in ['initial_failure', 'propagation', 'cascade']:
                        config_lines.append("circuit_breaker: enabled")
                        config_lines.append("bulkhead_isolation: enabled")
                        config_lines.append("timeout: 30s")
                        config_lines.append("fallback_service: enabled")
                        
                    else:
                        # Default patterns for unknown types
                        config_lines.append(f"timeout: {duration}s")
                        config_lines.append("monitoring: enabled")
                        config_lines.append("alerting: enabled")
                    
                    # Add more diverse target-specific config based on severity and spec type
                    config_lines.append(f"service_name: {target}")
                    config_lines.append(f"deployment.strategy: {spec_type}")
                    config_lines.append(f"failure.context: {target}_{spec_type}")
                    
                    # Add unique identifiers to ensure diversity
                    import hashlib
                    unique_id = hashlib.md5(f"{spec_type}_{target}_{severity}_{duration}".encode()).hexdigest()[:8]
                    config_lines.append(f"scenario.id: {unique_id}")
                    config_lines.append(f"pattern.fingerprint: {spec_type}_{severity}_{unique_id}")
                    
                    if severity == 'high':
                        config_lines.append(f"replicas: {5 + hash(target) % 3}")  # 5-7 replicas
                        config_lines.append(f"memory_limit: {1 + hash(target) % 3}GB")  # 1-3GB
                        config_lines.append(f"cpu_limit: {0.8 + (hash(target) % 3) * 0.1:.1f}")  # 0.8-1.0
                        config_lines.append(f"health_check_interval: {3 + hash(target) % 5}s")  # 3-7s
                        config_lines.append(f"error_rate_threshold: {0.01 + (hash(target) % 5) * 0.01:.3f}")
                    elif severity == 'medium':
                        config_lines.append(f"replicas: {2 + hash(target) % 3}")  # 2-4 replicas  
                        config_lines.append(f"memory_limit: {256 + (hash(target) % 3) * 256}MB")  # 256-768MB
                        config_lines.append(f"cpu_limit: {0.3 + (hash(target) % 3) * 0.1:.1f}")  # 0.3-0.5
                        config_lines.append(f"health_check_interval: {10 + hash(target) % 10}s")  # 10-19s
                        config_lines.append(f"timeout: {30 + (hash(target) % 5) * 10}s")  # 30-70s
                    else:
                        config_lines.append(f"replicas: {1 + hash(target) % 2}")  # 1-2 replicas
                        config_lines.append(f"memory_limit: {128 + (hash(target) % 3) * 64}MB")  # 128-256MB
                        config_lines.append(f"cpu_limit: {0.1 + (hash(target) % 3) * 0.1:.1f}")  # 0.1-0.3
                        config_lines.append(f"health_check_interval: {20 + hash(target) % 20}s")  # 20-39s
                        config_lines.append(f"connection_timeout: {5 + (hash(target) % 5) * 5}s")  # 5-25s
                    
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
        
        # Remove duplicates and sort
        unique_lines = sorted(set(config_lines))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Generated configuration lines for pattern matching validation\n")
            f.write(f"# Total lines: {len(unique_lines)}\n")
            f.write(f"# Generated at: {datetime.now().isoformat()}\n\n")
            
            for line in unique_lines:
                f.write(f"{line}\n")
        
        logger.info(f"📝 Written {len(unique_lines)} unique config lines for validation") 