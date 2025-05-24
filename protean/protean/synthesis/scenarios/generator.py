#!/usr/bin/env python3
"""
Scenario Generator with Phrase Template Integration
Part of Protean Pattern Discovery Engine - Week 1 Mission

Generates realistic infrastructure failure scenarios using real phrases
extracted from postmortem documents and DAGs.
"""

import random
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger


@dataclass
class InjectionSpec:
    """Specification for failure injection"""
    type: str
    target: str
    severity: str
    duration: int
    log_snippet: str
    metrics: Dict[str, Any]


@dataclass
class ScenarioEvent:
    """A single event in a scenario timeline"""
    timestamp: datetime
    event_type: str
    description: str
    injection_spec: Optional[InjectionSpec] = None


@dataclass
class Scenario:
    """A complete failure scenario"""
    category: str
    name: str
    description: str
    duration: int
    events: List[ScenarioEvent]
    metadata: Dict[str, Any]


class ScenarioGenerator:
    """Generate realistic infrastructure failure scenarios with phrase templates"""
    
    def __init__(self, phrase_templates_path: str = "data/synthetic/phrase_templates.yaml"):
        self.phrase_templates_path = Path(phrase_templates_path)
        self.phrase_templates = self._load_phrase_templates()
        self.scenario_categories = self._get_scenario_categories()
        
        # Infrastructure components for scenarios
        self.infrastructure_components = {
            'databases': ['mysql-primary', 'postgres-replica', 'redis-cache', 'mongodb-shard'],
            'services': ['user-service', 'payment-service', 'notification-service', 'auth-service'],
            'containers': ['web-frontend', 'api-backend', 'worker-consumer', 'nginx-proxy'],
            'clusters': ['prod-cluster', 'staging-cluster', 'worker-cluster'],
            'queues': ['payment-queue', 'notification-queue', 'batch-processing-queue'],
            'networks': ['vpc-main', 'subnet-private', 'load-balancer', 'api-gateway']
        }
        
        logger.info(f"🎭 ScenarioGenerator initialized with {len(self.phrase_templates)} pattern categories")
    
    def _load_phrase_templates(self) -> Dict[str, List[str]]:
        """Load phrase templates from YAML file"""
        try:
            if not self.phrase_templates_path.exists():
                logger.warning(f"⚠️  Phrase templates not found at {self.phrase_templates_path}")
                return {}
            
            with open(self.phrase_templates_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            templates = data.get('phrase_templates', {})
            logger.info(f"✅ Loaded phrase templates for {len(templates)} categories")
            return templates
            
        except Exception as e:
            logger.error(f"❌ Failed to load phrase templates: {e}")
            return {}
    
    def _get_scenario_categories(self) -> List[str]:
        """Get available scenario categories from phrase templates"""
        return list(self.phrase_templates.keys()) if self.phrase_templates else [
            'CircuitBreaker', 'CloudNative', 'ConfigurationDrift', 'DatabaseFailure',
            'DeploymentFailure', 'EventDriven', 'LoadBalancingIssue', 'CascadingFailure'
        ]
    
    def _choose_random_phrase(self, category: str) -> str:
        """Choose a random phrase from template matching scenario category"""
        if category in self.phrase_templates and self.phrase_templates[category]:
            phrase = random.choice(self.phrase_templates[category])
            logger.debug(f"🎯 Selected phrase for {category}: '{phrase}'")
            return phrase
        else:
            # Fallback to generic infrastructure phrases
            fallback_phrases = [
                "service unavailable", "connection timeout", "memory limit exceeded",
                "disk space full", "network unreachable", "authentication failed",
                "rate limit exceeded", "circuit breaker triggered", "deployment failed",
                "container crashed", "database connection lost", "queue overflow"
            ]
            phrase = random.choice(fallback_phrases)
            logger.debug(f"🔄 Using fallback phrase for {category}: '{phrase}'")
            return phrase
    
    def _generate_timeline(self, category: str, duration: int) -> List[ScenarioEvent]:
        """Generate a timeline of events for a scenario"""
        events = []
        start_time = datetime.now()
        
        # Initial trigger event
        trigger_phrase = self._choose_random_phrase(category)
        events.append(ScenarioEvent(
            timestamp=start_time,
            event_type="trigger",
            description=f"Initial failure detected: {trigger_phrase}",
            injection_spec=None
        ))
        
        # Generate failure progression events
        num_events = random.randint(3, 8)
        for i in range(num_events):
            event_time = start_time + timedelta(
                seconds=random.randint(10, duration // num_events)
            )
            
            # Choose event type based on category
            event_type = self._choose_event_type(category)
            phrase = self._choose_random_phrase(category)
            
            # Create injection spec for failure events
            injection_spec = None
            if event_type in ['failure', 'degradation', 'timeout']:
                injection_spec = self._create_injection_spec(category, phrase, event_type)
            
            events.append(ScenarioEvent(
                timestamp=event_time,
                event_type=event_type,
                description=f"{event_type.title()}: {phrase}",
                injection_spec=injection_spec
            ))
        
        # Recovery event
        recovery_phrase = self._choose_recovery_phrase(category)
        events.append(ScenarioEvent(
            timestamp=start_time + timedelta(seconds=duration),
            event_type="recovery",
            description=f"System recovered: {recovery_phrase}",
            injection_spec=None
        ))
        
        return sorted(events, key=lambda x: x.timestamp)
    
    def _choose_event_type(self, category: str) -> str:
        """Choose event type based on scenario category"""
        event_types = {
            'CircuitBreaker': ['timeout', 'failure', 'recovery', 'degradation'],
            'CloudNative': ['pod_restart', 'scaling', 'deployment', 'failure'],
            'ConfigurationDrift': ['config_change', 'restart', 'failure', 'rollback'],
            'DatabaseFailure': ['connection_loss', 'timeout', 'corruption', 'recovery'],
            'DeploymentFailure': ['deployment', 'rollback', 'failure', 'scaling'],
            'EventDriven': ['queue_overflow', 'message_loss', 'timeout', 'recovery'],
            'LoadBalancingIssue': ['traffic_spike', 'backend_failure', 'timeout', 'scaling'],
            'CascadingFailure': ['initial_failure', 'propagation', 'cascade', 'recovery']
        }
        
        return random.choice(event_types.get(category, ['failure', 'timeout', 'recovery']))
    
    def _choose_recovery_phrase(self, category: str) -> str:
        """Choose appropriate recovery phrase for the category"""
        recovery_phrases = {
            'CircuitBreaker': 'circuit breaker reset',
            'CloudNative': 'pod restarted successfully',
            'ConfigurationDrift': 'configuration restored',
            'DatabaseFailure': 'database connection restored',
            'DeploymentFailure': 'deployment completed',
            'EventDriven': 'message queue drained',
            'LoadBalancingIssue': 'load balanced restored',
            'CascadingFailure': 'all services recovered'
        }
        
        return recovery_phrases.get(category, 'system restored')
    
    def _create_injection_spec(self, category: str, phrase: str, event_type: str) -> InjectionSpec:
        """Create failure injection specification with embedded phrase"""
        # Choose target component based on category
        target = self._choose_target_component(category)
        
        # Determine severity based on phrase content
        severity = self._determine_severity(phrase, event_type)
        
        # Create realistic log snippet with embedded phrase
        log_snippet = self._create_log_snippet(phrase, target, event_type)
        
        # Generate relevant metrics
        metrics = self._generate_metrics(category, severity)
        
        return InjectionSpec(
            type=event_type,
            target=target,
            severity=severity,
            duration=random.randint(30, 300),
            log_snippet=log_snippet,
            metrics=metrics
        )
    
    def _choose_target_component(self, category: str) -> str:
        """Choose target component based on scenario category"""
        category_mappings = {
            'CircuitBreaker': 'services',
            'CloudNative': 'containers',
            'ConfigurationDrift': 'services',
            'DatabaseFailure': 'databases',
            'DeploymentFailure': 'containers',
            'EventDriven': 'queues',
            'LoadBalancingIssue': 'networks',
            'CascadingFailure': 'services'
        }
        
        component_type = category_mappings.get(category, 'services')
        return random.choice(self.infrastructure_components[component_type])
    
    def _determine_severity(self, phrase: str, event_type: str) -> str:
        """Determine failure severity based on phrase content and event type"""
        high_severity_indicators = [
            'crash', 'failed', 'critical', 'fatal', 'emergency', 'down', 'unavailable',
            'timeout', 'refused', 'lost', 'exceeded', 'overflow', 'corruption'
        ]
        
        medium_severity_indicators = [
            'slow', 'degraded', 'warning', 'retry', 'limit', 'queue', 'delay'
        ]
        
        phrase_lower = phrase.lower()
        
        if any(indicator in phrase_lower for indicator in high_severity_indicators):
            return 'high'
        elif any(indicator in phrase_lower for indicator in medium_severity_indicators):
            return 'medium'
        elif event_type in ['failure', 'timeout', 'crash']:
            return 'high'
        else:
            return 'low'
    
    def _create_log_snippet(self, phrase: str, target: str, event_type: str) -> str:
        """Create realistic log snippet with embedded phrase"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Choose log format based on component type
        if 'service' in target:
            return f"[{timestamp}] ERROR {target}: {phrase} - operation failed"
        elif 'container' in target or 'pod' in target:
            return f"[{timestamp}] WARN {target}: container experiencing {phrase}"
        elif 'database' in target or 'mysql' in target or 'postgres' in target:
            return f"[{timestamp}] FATAL {target}: database {phrase} detected"
        elif 'queue' in target:
            return f"[{timestamp}] ERROR {target}: queue processing {phrase}"
        elif 'cluster' in target:
            return f"[{timestamp}] CRIT {target}: cluster node {phrase}"
        else:
            return f"[{timestamp}] ERROR {target}: {phrase} in {event_type}"
    
    def _generate_metrics(self, category: str, severity: str) -> Dict[str, Any]:
        """Generate relevant metrics for the failure scenario"""
        base_metrics = {
            'error_rate': random.uniform(0.1, 0.9) if severity == 'high' else random.uniform(0.01, 0.3),
            'response_time_ms': random.randint(1000, 10000) if severity == 'high' else random.randint(100, 2000),
            'cpu_usage': random.uniform(0.7, 1.0) if severity == 'high' else random.uniform(0.3, 0.8),
            'memory_usage': random.uniform(0.8, 1.0) if severity == 'high' else random.uniform(0.4, 0.7)
        }
        
        # Add category-specific metrics
        if category == 'DatabaseFailure':
            base_metrics.update({
                'connection_pool_usage': random.uniform(0.8, 1.0),
                'query_time_ms': random.randint(5000, 30000),
                'deadlocks_per_sec': random.randint(1, 10)
            })
        elif category == 'EventDriven':
            base_metrics.update({
                'queue_depth': random.randint(1000, 50000),
                'message_processing_rate': random.uniform(0.1, 10.0),
                'consumer_lag_ms': random.randint(10000, 300000)
            })
        elif category == 'LoadBalancingIssue':
            base_metrics.update({
                'requests_per_sec': random.randint(100, 10000),
                'backend_health_pct': random.uniform(0.3, 0.8),
                'connection_timeouts': random.randint(10, 1000)
            })
        
        return base_metrics
    
    def generate_scenario(self, category: Optional[str] = None, duration: int = 600) -> Scenario:
        """Generate a complete failure scenario"""
        if category is None:
            category = random.choice(self.scenario_categories)
        
        if category not in self.scenario_categories:
            logger.warning(f"⚠️  Unknown category '{category}', using random category")
            category = random.choice(self.scenario_categories)
        
        # Generate scenario metadata
        scenario_name = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        description = self._generate_scenario_description(category)
        
        # Generate timeline events
        events = self._generate_timeline(category, duration)
        
        scenario = Scenario(
            category=category,
            name=scenario_name,
            description=description,
            duration=duration,
            events=events,
            metadata={
                'generated_at': datetime.now().isoformat(),
                'phrase_templates_used': len(self.phrase_templates.get(category, [])),
                'total_events': len(events),
                'severity_distribution': self._calculate_severity_distribution(events)
            }
        )
        
        logger.info(f"🎭 Generated scenario '{scenario_name}' with {len(events)} events")
        return scenario
    
    def _generate_scenario_description(self, category: str) -> str:
        """Generate descriptive text for the scenario"""
        descriptions = {
            'CircuitBreaker': 'Circuit breaker activation due to cascading service failures',
            'CloudNative': 'Kubernetes pod and container orchestration issues',
            'ConfigurationDrift': 'Configuration inconsistencies causing service degradation',
            'DatabaseFailure': 'Database connectivity and performance issues',
            'DeploymentFailure': 'Application deployment and rollback scenarios',
            'EventDriven': 'Message queue and event processing failures',
            'LoadBalancingIssue': 'Load balancer and traffic distribution problems',
            'CascadingFailure': 'Multi-service failure propagation scenario'
        }
        
        return descriptions.get(category, f"Infrastructure failure scenario for {category}")
    
    def _calculate_severity_distribution(self, events: List[ScenarioEvent]) -> Dict[str, int]:
        """Calculate distribution of severity levels in the scenario"""
        distribution = {'low': 0, 'medium': 0, 'high': 0}
        
        for event in events:
            if event.injection_spec:
                severity = event.injection_spec.severity
                distribution[severity] = distribution.get(severity, 0) + 1
        
        return distribution
    
    def generate_batch_scenarios(self, count: int = 10, categories: Optional[List[str]] = None) -> List[Scenario]:
        """Generate a batch of scenarios for testing"""
        if categories is None:
            categories = self.scenario_categories
        
        scenarios = []
        for i in range(count):
            category = random.choice(categories)
            duration = random.randint(300, 1800)  # 5-30 minutes
            scenario = self.generate_scenario(category, duration)
            scenarios.append(scenario)
        
        logger.info(f"🎭 Generated batch of {len(scenarios)} scenarios")
        return scenarios


def main():
    """Example usage and testing"""
    try:
        generator = ScenarioGenerator()
        
        # Generate a single scenario
        scenario = generator.generate_scenario('DatabaseFailure', duration=900)
        
        print(f"Generated Scenario: {scenario.name}")
        print(f"Category: {scenario.category}")
        print(f"Description: {scenario.description}")
        print(f"Duration: {scenario.duration}s")
        print(f"Events: {len(scenario.events)}")
        
        print("\nTimeline:")
        for event in scenario.events[:5]:  # Show first 5 events
            print(f"  {event.timestamp.strftime('%H:%M:%S')} - {event.event_type}: {event.description}")
            if event.injection_spec:
                print(f"    → Target: {event.injection_spec.target}")
                print(f"    → Log: {event.injection_spec.log_snippet}")
        
        # Generate batch scenarios
        batch = generator.generate_batch_scenarios(count=5)
        print(f"\nGenerated {len(batch)} scenarios in batch")
        
    except Exception as e:
        logger.error(f"❌ Scenario generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 