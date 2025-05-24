"""
Graph Transformation Operations
Core operations for pattern evolution and transformation
"""

from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import uuid
from copy import deepcopy

from .graph import PatternGraph, Node, Edge
from .cost import CostModel, OperationCost, CostType


class GraphOperation(ABC):
    """Abstract base class for graph operations"""
    
    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()
        self.operation_id = str(uuid.uuid4())[:8]
        self.applied_cost: Optional[OperationCost] = None
    
    @property
    @abstractmethod
    def op_type(self) -> str:
        """Operation type identifier"""
        pass
    
    @abstractmethod
    def apply(self, graph: PatternGraph) -> PatternGraph:
        """Apply operation to graph"""
        pass
    
    @abstractmethod
    def inverse(self, graph: PatternGraph) -> PatternGraph:
        """Apply inverse operation to graph"""
        pass
    
    def calculate_cost(self, graph: PatternGraph, **kwargs) -> OperationCost:
        """Calculate cost for this operation on the given graph"""
        return self.cost_model.calculate_operation_cost(
            op_type=self.op_type,
            **kwargs
        )


class Replicate(GraphOperation):
    """
    Replicate operation: Duplicates a node and its connections
    Creates count copies of the specified node
    """
    
    def __init__(self, node_id: str, count: int = 2, cost_model: Optional[CostModel] = None):
        super().__init__(cost_model)
        self.node_id = node_id
        self.count = count
        self.created_node_ids: List[str] = []
    
    @property
    def op_type(self) -> str:
        return "Replicate"
    
    def apply(self, graph: PatternGraph) -> PatternGraph:
        """Apply replication operation"""
        result_graph = graph.copy()
        
        # Check if node exists
        if not result_graph.has_node(self.node_id):
            raise ValueError(f"Node {self.node_id} not found in graph")
        
        original_node = result_graph.get_node(self.node_id)
        if not original_node:
            raise ValueError(f"Could not retrieve node {self.node_id}")
        
        # Get all incoming and outgoing edges for the original node
        incoming_edges = []
        outgoing_edges = []
        
        for edge in result_graph.get_edges():
            if edge.target == self.node_id:
                incoming_edges.append(edge)
            elif edge.source == self.node_id:
                outgoing_edges.append(edge)
        
        # Create replicated nodes
        self.created_node_ids = []
        for i in range(self.count):
            # Create new node with unique ID
            new_node_id = f"{self.node_id}_replica_{i+1}"
            new_node = Node(
                node_id=new_node_id,
                node_type=original_node.node_type,
                attributes=deepcopy(original_node.attributes),
                cost=original_node.cost
            )
            result_graph.add_node(new_node)
            self.created_node_ids.append(new_node_id)
            
            # Replicate incoming edges
            for edge in incoming_edges:
                new_edge = Edge(
                    source=edge.source,
                    target=new_node_id,
                    edge_type=edge.edge_type,
                    attributes=deepcopy(edge.attributes),
                    weight=edge.weight
                )
                result_graph.add_edge(new_edge)
            
            # Replicate outgoing edges
            for edge in outgoing_edges:
                new_edge = Edge(
                    source=new_node_id,
                    target=edge.target,
                    edge_type=edge.edge_type,
                    attributes=deepcopy(edge.attributes),
                    weight=edge.weight
                )
                result_graph.add_edge(new_edge)
        
        # Calculate and record cost
        edge_count_delta = len(incoming_edges + outgoing_edges) * self.count
        self.applied_cost = self.calculate_cost(
            graph, 
            node_count_delta=self.count,
            edge_count_delta=edge_count_delta,
            metadata={'replicated_node': self.node_id, 'count': self.count}
        )
        self.cost_model.record_operation(self.applied_cost)
        
        return result_graph
    
    def inverse(self, graph: PatternGraph) -> PatternGraph:
        """Remove replicated nodes (inverse operation)"""
        result_graph = graph.copy()
        
        # Remove created nodes and their edges
        for node_id in self.created_node_ids:
            if result_graph.has_node(node_id):
                result_graph.remove_node(node_id)
        
        # Calculate inverse cost (should be negative of original)
        if self.applied_cost:
            inverse_cost = OperationCost(
                op_type=f"Inverse{self.op_type}",
                cost_delta=-self.applied_cost.cost_delta,
                cost_breakdown={
                    cost_type: -cost 
                    for cost_type, cost in self.applied_cost.cost_breakdown.items()
                },
                metadata={'inverse_of': self.operation_id}
            )
            self.cost_model.record_operation(inverse_cost)
        
        return result_graph


class CircuitBreaker(GraphOperation):
    """
    Circuit Breaker operation: Adds circuit breaker pattern around a node
    Introduces failure detection and fallback mechanisms
    """
    
    def __init__(self, node_id: str, timeout: float = 5.0, 
                 failure_threshold: int = 3, cost_model: Optional[CostModel] = None):
        super().__init__(cost_model)
        self.node_id = node_id
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.breaker_node_id: Optional[str] = None
        self.fallback_node_id: Optional[str] = None
        self.monitor_node_id: Optional[str] = None
        self.original_edges: List[Edge] = []  # Store original edges for restoration
    
    @property
    def op_type(self) -> str:
        return "CircuitBreaker"
    
    def apply(self, graph: PatternGraph) -> PatternGraph:
        """Apply circuit breaker pattern"""
        result_graph = graph.copy()
        
        # Check if node exists
        if not result_graph.has_node(self.node_id):
            raise ValueError(f"Node {self.node_id} not found in graph")
        
        original_node = result_graph.get_node(self.node_id)
        if not original_node:
            raise ValueError(f"Could not retrieve node {self.node_id}")
        
        # Create circuit breaker components
        self.breaker_node_id = f"{self.node_id}_circuit_breaker"
        self.fallback_node_id = f"{self.node_id}_fallback"
        self.monitor_node_id = f"{self.node_id}_monitor"
        
        # Circuit breaker control node
        breaker_node = Node(
            node_id=self.breaker_node_id,
            node_type="CircuitBreakerController",
            attributes={
                "timeout": self.timeout,
                "failure_threshold": self.failure_threshold,
                "protected_node": self.node_id
            },
            cost=1.5
        )
        result_graph.add_node(breaker_node)
        
        # Fallback node
        fallback_node = Node(
            node_id=self.fallback_node_id,
            node_type="FallbackHandler",
            attributes={
                "fallback_strategy": "default",
                "protected_node": self.node_id
            },
            cost=1.0
        )
        result_graph.add_node(fallback_node)
        
        # Monitor node
        monitor_node = Node(
            node_id=self.monitor_node_id,
            node_type="HealthMonitor",
            attributes={
                "check_interval": 1.0,
                "monitored_node": self.node_id
            },
            cost=0.5
        )
        result_graph.add_node(monitor_node)
        
        # Get original edges to reroute through circuit breaker
        incoming_edges = []
        for edge in result_graph.get_edges():
            if edge.target == self.node_id:
                incoming_edges.append(edge)
        
        # Store original edges for restoration
        self.original_edges = [Edge(
            source=edge.source,
            target=edge.target,
            edge_type=edge.edge_type,
            attributes=deepcopy(edge.attributes),
            weight=edge.weight
        ) for edge in incoming_edges]
        
        # Reroute incoming edges through circuit breaker
        for edge in incoming_edges:
            # Remove original edge
            result_graph.remove_edge(edge.source, edge.target)
            
            # Add edge to circuit breaker
            breaker_edge = Edge(
                source=edge.source,
                target=self.breaker_node_id,
                edge_type="circuit_protected",
                attributes=deepcopy(edge.attributes),
                weight=edge.weight
            )
            result_graph.add_edge(breaker_edge)
        
        # Add circuit breaker internal connections
        # Breaker -> Original Node
        result_graph.add_edge(Edge(
            source=self.breaker_node_id,
            target=self.node_id,
            edge_type="protected_call",
            weight=1.0
        ))
        
        # Breaker -> Fallback
        result_graph.add_edge(Edge(
            source=self.breaker_node_id,
            target=self.fallback_node_id,
            edge_type="fallback_trigger",
            weight=1.0
        ))
        
        # Monitor -> Breaker
        result_graph.add_edge(Edge(
            source=self.monitor_node_id,
            target=self.breaker_node_id,
            edge_type="health_check",
            weight=0.5
        ))
        
        # Monitor -> Original Node
        result_graph.add_edge(Edge(
            source=self.monitor_node_id,
            target=self.node_id,
            edge_type="monitoring",
            weight=0.5
        ))
        
        # Calculate and record cost
        nodes_added = 3  # breaker, fallback, monitor
        edges_added = 4 + len(incoming_edges)  # internal edges + rerouted edges
        
        self.applied_cost = self.calculate_cost(
            graph,
            node_count_delta=nodes_added,
            edge_count_delta=edges_added,
            complexity_factor=1.5,  # Circuit breaker adds complexity
            metadata={
                'protected_node': self.node_id,
                'timeout': self.timeout,
                'failure_threshold': self.failure_threshold
            }
        )
        self.cost_model.record_operation(self.applied_cost)
        
        return result_graph
    
    def inverse(self, graph: PatternGraph) -> PatternGraph:
        """Remove circuit breaker pattern (inverse operation)"""
        result_graph = graph.copy()
        
        # Remove circuit breaker nodes (this will automatically remove their edges)
        nodes_to_remove = [
            self.breaker_node_id,
            self.fallback_node_id, 
            self.monitor_node_id
        ]
        
        for node_id in nodes_to_remove:
            if node_id and result_graph.has_node(node_id):
                result_graph.remove_node(node_id)
        
        # Restore original edges exactly as they were
        for edge in self.original_edges:
            result_graph.add_edge(edge)
        
        # Calculate inverse cost
        if self.applied_cost:
            inverse_cost = OperationCost(
                op_type=f"Inverse{self.op_type}",
                cost_delta=-self.applied_cost.cost_delta,
                cost_breakdown={
                    cost_type: -cost 
                    for cost_type, cost in self.applied_cost.cost_breakdown.items()
                },
                metadata={'inverse_of': self.operation_id}
            )
            self.cost_model.record_operation(inverse_cost)
        
        return result_graph


class Cache(GraphOperation):
    """
    Cache operation: Adds caching layer to a node
    Can reduce overall system cost (negative cost delta allowed)
    """
    
    def __init__(self, node_id: str, cache_type: str = "memory", 
                 ttl: float = 300.0, cost_model: Optional[CostModel] = None):
        super().__init__(cost_model)
        self.node_id = node_id
        self.cache_type = cache_type
        self.ttl = ttl
        self.cache_node_id: Optional[str] = None
    
    @property
    def op_type(self) -> str:
        return "Cache"
    
    def apply(self, graph: PatternGraph) -> PatternGraph:
        """Apply caching pattern"""
        result_graph = graph.copy()
        
        if not result_graph.has_node(self.node_id):
            raise ValueError(f"Node {self.node_id} not found in graph")
        
        # Create cache node
        self.cache_node_id = f"{self.node_id}_cache"
        cache_node = Node(
            node_id=self.cache_node_id,
            node_type="CacheLayer",
            attributes={
                "cache_type": self.cache_type,
                "ttl": self.ttl,
                "cached_node": self.node_id
            },
            cost=2.0  # Cache uses memory but saves compute
        )
        result_graph.add_node(cache_node)
        
        # Add cache relationship
        result_graph.add_edge(Edge(
            source=self.cache_node_id,
            target=self.node_id,
            edge_type="cache_backing",
            weight=0.5
        ))
        
        # Calculate cost (can be negative for Cache operations)
        self.applied_cost = self.calculate_cost(
            graph,
            node_count_delta=1,
            edge_count_delta=1,
            metadata={'cached_node': self.node_id, 'cache_type': self.cache_type}
        )
        self.cost_model.record_operation(self.applied_cost)
        
        return result_graph
    
    def inverse(self, graph: PatternGraph) -> PatternGraph:
        """Remove cache (inverse operation)"""
        result_graph = graph.copy()
        
        if self.cache_node_id and result_graph.has_node(self.cache_node_id):
            result_graph.remove_node(self.cache_node_id)
        
        # Calculate inverse cost
        if self.applied_cost:
            inverse_cost = OperationCost(
                op_type=f"Inverse{self.op_type}",
                cost_delta=-self.applied_cost.cost_delta,
                cost_breakdown={
                    cost_type: -cost 
                    for cost_type, cost in self.applied_cost.cost_breakdown.items()
                },
                metadata={'inverse_of': self.operation_id}
            )
            self.cost_model.record_operation(inverse_cost)
        
        return result_graph


class Split(GraphOperation):
    """Split operation: Splits a node into multiple specialized nodes"""
    
    def __init__(self, node_id: str, split_strategies: List[str], 
                 cost_model: Optional[CostModel] = None):
        super().__init__(cost_model)
        self.node_id = node_id
        self.split_strategies = split_strategies
        self.split_node_ids: List[str] = []
    
    @property
    def op_type(self) -> str:
        return "Split"
    
    def apply(self, graph: PatternGraph) -> PatternGraph:
        """Apply split operation"""
        result_graph = graph.copy()
        
        if not result_graph.has_node(self.node_id):
            raise ValueError(f"Node {self.node_id} not found in graph")
        
        original_node = result_graph.get_node(self.node_id)
        
        # Create split nodes
        self.split_node_ids = []
        for i, strategy in enumerate(self.split_strategies):
            split_id = f"{self.node_id}_split_{strategy}"
            split_node = Node(
                node_id=split_id,
                node_type=f"{original_node.node_type}_{strategy}",
                attributes={**original_node.attributes, "split_strategy": strategy},
                cost=original_node.cost * 0.7  # Specialized nodes are more efficient
            )
            result_graph.add_node(split_node)
            self.split_node_ids.append(split_id)
        
        # Remove original node
        result_graph.remove_node(self.node_id)
        
        self.applied_cost = self.calculate_cost(
            graph,
            node_count_delta=len(self.split_strategies) - 1,
            metadata={'split_node': self.node_id, 'strategies': self.split_strategies}
        )
        self.cost_model.record_operation(self.applied_cost)
        
        return result_graph
    
    def inverse(self, graph: PatternGraph) -> PatternGraph:
        """Merge split nodes back (inverse operation)"""
        result_graph = graph.copy()
        
        # Remove split nodes
        for node_id in self.split_node_ids:
            if result_graph.has_node(node_id):
                result_graph.remove_node(node_id)
        
        # Restore original node (simplified)
        if not result_graph.has_node(self.node_id):
            original_node = Node(
                node_id=self.node_id,
                node_type="RestoredNode",
                cost=1.0
            )
            result_graph.add_node(original_node)
        
        if self.applied_cost:
            inverse_cost = OperationCost(
                op_type=f"Inverse{self.op_type}",
                cost_delta=-self.applied_cost.cost_delta,
                cost_breakdown={
                    cost_type: -cost 
                    for cost_type, cost in self.applied_cost.cost_breakdown.items()
                },
                metadata={'inverse_of': self.operation_id}
            )
            self.cost_model.record_operation(inverse_cost)
        
        return result_graph


class Merge(GraphOperation):
    """Merge operation: Combines multiple nodes into one"""
    
    def __init__(self, node_ids: List[str], merged_type: str = "MergedNode",
                 cost_model: Optional[CostModel] = None):
        super().__init__(cost_model)
        self.node_ids = node_ids
        self.merged_type = merged_type
        self.merged_node_id: Optional[str] = None
        self.original_nodes: List[Node] = []
    
    @property
    def op_type(self) -> str:
        return "Merge"
    
    def apply(self, graph: PatternGraph) -> PatternGraph:
        """Apply merge operation"""
        result_graph = graph.copy()
        
        # Store original nodes for inverse operation
        self.original_nodes = []
        for node_id in self.node_ids:
            if not result_graph.has_node(node_id):
                raise ValueError(f"Node {node_id} not found in graph")
            node = result_graph.get_node(node_id)
            if node:
                self.original_nodes.append(node)
        
        # Create merged node
        self.merged_node_id = f"merged_{'_'.join(self.node_ids)}"
        merged_attributes = {}
        total_cost = 0.0
        
        for node in self.original_nodes:
            merged_attributes.update(node.attributes)
            total_cost += node.cost
        
        merged_node = Node(
            node_id=self.merged_node_id,
            node_type=self.merged_type,
            attributes=merged_attributes,
            cost=total_cost * 0.8  # Merged node is more efficient
        )
        result_graph.add_node(merged_node)
        
        # Remove original nodes
        for node_id in self.node_ids:
            result_graph.remove_node(node_id)
        
        self.applied_cost = self.calculate_cost(
            graph,
            node_count_delta=1 - len(self.node_ids),
            metadata={'merged_nodes': self.node_ids}
        )
        self.cost_model.record_operation(self.applied_cost)
        
        return result_graph
    
    def inverse(self, graph: PatternGraph) -> PatternGraph:
        """Split merged node back (inverse operation)"""
        result_graph = graph.copy()
        
        # Remove merged node
        if self.merged_node_id and result_graph.has_node(self.merged_node_id):
            result_graph.remove_node(self.merged_node_id)
        
        # Restore original nodes
        for node in self.original_nodes:
            result_graph.add_node(node)
        
        if self.applied_cost:
            inverse_cost = OperationCost(
                op_type=f"Inverse{self.op_type}",
                cost_delta=-self.applied_cost.cost_delta,
                cost_breakdown={
                    cost_type: -cost 
                    for cost_type, cost in self.applied_cost.cost_breakdown.items()
                },
                metadata={'inverse_of': self.operation_id}
            )
            self.cost_model.record_operation(inverse_cost)
        
        return result_graph 