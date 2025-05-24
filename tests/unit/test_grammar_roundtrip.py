"""
Round-trip Tests for Grammar Primitives
Tests that operations can be applied and inverted while maintaining graph isomorphism
and cost constraints.
"""

import pytest
from protean.protean.grammar.primitives import (
    PatternGraph, Node, Edge, 
    Replicate, CircuitBreaker, Cache, Split, Merge,
    CostModel, CostType
)
from protean.protean.grammar.primitives.graph import create_simple_graph


class TestGrammarRoundTrip:
    """Test suite for round-trip operations"""
    
    def setup_method(self):
        """Setup for each test"""
        self.cost_model = CostModel()
    
    def test_simple_graph_creation(self):
        """Test that we can create the basic A→B graph"""
        graph = create_simple_graph()
        
        # Verify structure
        assert graph.node_count == 2
        assert graph.edge_count == 1
        assert graph.has_node("A")
        assert graph.has_node("B")
        assert graph.has_edge("A", "B")
        
        # Verify node types
        node_a = graph.get_node("A")
        node_b = graph.get_node("B")
        assert node_a.node_type == "A"
        assert node_b.node_type == "B"
        
        # Verify edge
        edge = graph.get_edge("A", "B")
        assert edge.edge_type == "dependency"
    
    def test_replicate_roundtrip(self):
        """Test Replicate operation and its inverse"""
        # Build toy graph A→B
        original_graph = create_simple_graph()
        original_cost = original_graph.calculate_cost()
        
        # Apply Replicate(node_id='B', count=2)
        replicate_op = Replicate(node_id='B', count=2, cost_model=self.cost_model)
        
        # Apply operation
        replicated_graph = replicate_op.apply(original_graph)
        
        # Verify replication worked
        assert replicated_graph.node_count == 4  # A + B + 2 replicas
        assert replicated_graph.has_node("A")
        assert replicated_graph.has_node("B")
        assert replicated_graph.has_node("B_replica_1")
        assert replicated_graph.has_node("B_replica_2")
        
        # Verify edges to replicas
        assert replicated_graph.has_edge("A", "B_replica_1")
        assert replicated_graph.has_edge("A", "B_replica_2")
        
        # Verify cost constraint (should be positive for non-Cache operations)
        assert replicate_op.applied_cost.cost_delta > 0
        assert replicate_op.applied_cost.validate()
        
        # Apply inverse
        restored_graph = replicate_op.inverse(replicated_graph)
        
        # Verify round-trip: final graph should be isomorphic to original
        assert restored_graph.is_isomorphic(original_graph)
        assert restored_graph.node_count == original_graph.node_count
        assert restored_graph.edge_count == original_graph.edge_count
    
    def test_circuit_breaker_roundtrip(self):
        """Test CircuitBreaker operation and its inverse"""
        # Build toy graph A→B
        original_graph = create_simple_graph()
        
        # Apply CircuitBreaker to node B
        circuit_breaker_op = CircuitBreaker(
            node_id='B', 
            timeout=5.0, 
            failure_threshold=3,
            cost_model=self.cost_model
        )
        
        # Apply operation
        protected_graph = circuit_breaker_op.apply(original_graph)
        
        # Verify circuit breaker components were added
        assert protected_graph.node_count == 5  # A + B + breaker + fallback + monitor
        assert protected_graph.has_node("B_circuit_breaker")
        assert protected_graph.has_node("B_fallback")
        assert protected_graph.has_node("B_monitor")
        
        # Verify original edge was rerouted through circuit breaker
        assert not protected_graph.has_edge("A", "B")  # Original edge removed
        assert protected_graph.has_edge("A", "B_circuit_breaker")  # Rerouted
        assert protected_graph.has_edge("B_circuit_breaker", "B")  # Protected call
        
        # Verify cost constraint (should be positive for non-Cache operations)
        assert circuit_breaker_op.applied_cost.cost_delta > 0
        assert circuit_breaker_op.applied_cost.validate()
        
        # Apply inverse
        restored_graph = circuit_breaker_op.inverse(protected_graph)
        
        # Verify round-trip: final graph should be isomorphic to original
        assert restored_graph.is_isomorphic(original_graph)
        assert restored_graph.node_count == original_graph.node_count
        assert restored_graph.edge_count == original_graph.edge_count
    
    def test_cache_operation_negative_cost_allowed(self):
        """Test that Cache operations can have negative cost (cost-saving)"""
        original_graph = create_simple_graph()
        
        # Apply Cache operation
        cache_op = Cache(node_id='B', cache_type='memory', cost_model=self.cost_model)
        cached_graph = cache_op.apply(original_graph)
        
        # Cache operations are allowed to have negative cost delta
        # (they save compute cost but use memory)
        assert cache_op.applied_cost.op_type == 'Cache'
        assert cache_op.applied_cost.validate()  # Should pass validation
        
        # Apply inverse
        restored_graph = cache_op.inverse(cached_graph)
        
        # Verify round-trip
        assert restored_graph.is_isomorphic(original_graph)
    
    def test_combined_operations_roundtrip(self):
        """Test combined operations: Replicate then CircuitBreaker, then inverse both"""
        # Build toy graph A→B
        original_graph = create_simple_graph()
        
        # Step 1: Apply Replicate(node_id='B', count=2)
        replicate_op = Replicate(node_id='B', count=2, cost_model=self.cost_model)
        step1_graph = replicate_op.apply(original_graph)
        
        # Step 2: Apply CircuitBreaker to original node B
        circuit_breaker_op = CircuitBreaker(
            node_id='B', 
            timeout=10.0, 
            failure_threshold=5,
            cost_model=self.cost_model
        )
        step2_graph = circuit_breaker_op.apply(step1_graph)
        
        # Verify both operations applied
        assert step2_graph.node_count > original_graph.node_count
        assert step2_graph.has_node("B_replica_1")
        assert step2_graph.has_node("B_replica_2")
        assert step2_graph.has_node("B_circuit_breaker")
        
        # Step 3: Apply inverse operations in reverse order
        # First inverse CircuitBreaker
        step3_graph = circuit_breaker_op.inverse(step2_graph)
        
        # Then inverse Replicate
        final_graph = replicate_op.inverse(step3_graph)
        
        # Verify complete round-trip: final graph should be isomorphic to original
        assert final_graph.is_isomorphic(original_graph)
        assert final_graph.node_count == original_graph.node_count
        assert final_graph.edge_count == original_graph.edge_count
    
    def test_cost_constraint_enforcement(self):
        """Test that cost constraints are properly enforced"""
        original_graph = create_simple_graph()
        
        # Test that non-Cache operations cannot have negative cost
        replicate_op = Replicate(node_id='B', count=2, cost_model=self.cost_model)
        replicated_graph = replicate_op.apply(original_graph)
        
        # Verify cost is positive for Replicate
        assert replicate_op.applied_cost.cost_delta > 0
        
        # Test CircuitBreaker cost
        circuit_breaker_op = CircuitBreaker(node_id='B', cost_model=self.cost_model)
        protected_graph = circuit_breaker_op.apply(original_graph)
        
        # Verify cost is positive for CircuitBreaker
        assert circuit_breaker_op.applied_cost.cost_delta > 0
        
        # Test that Cache can have negative cost (this is allowed)
        cache_op = Cache(node_id='B', cost_model=self.cost_model)
        cached_graph = cache_op.apply(original_graph)
        
        # Cache may have negative cost delta (saves compute, uses memory)
        assert cache_op.applied_cost.op_type == 'Cache'
        assert cache_op.applied_cost.validate()
    
    def test_cost_tracking_accuracy(self):
        """Test that cost tracking is accurate across operations"""
        original_graph = create_simple_graph()
        cost_model = CostModel()
        
        # Track costs through multiple operations
        initial_cost = cost_model.get_total_cost()
        assert initial_cost == 0.0
        
        # Apply Replicate
        replicate_op = Replicate(node_id='B', count=2, cost_model=cost_model)
        replicated_graph = replicate_op.apply(original_graph)
        
        after_replicate_cost = cost_model.get_total_cost()
        assert after_replicate_cost > initial_cost
        
        # Apply CircuitBreaker
        circuit_breaker_op = CircuitBreaker(node_id='B', cost_model=cost_model)
        protected_graph = circuit_breaker_op.apply(replicated_graph)
        
        after_circuit_breaker_cost = cost_model.get_total_cost()
        assert after_circuit_breaker_cost > after_replicate_cost
        
        # Apply inverses
        step1_graph = circuit_breaker_op.inverse(protected_graph)
        after_inverse_cb_cost = cost_model.get_total_cost()
        
        step2_graph = replicate_op.inverse(step1_graph)
        final_cost = cost_model.get_total_cost()
        
        # After all inverses, cost should be close to zero (within small tolerance)
        assert abs(final_cost) < 0.1
        
        # Verify cost constraint validation
        assert cost_model.validate_cost_constraints()
    
    def test_graph_isomorphism_precision(self):
        """Test that isomorphism check is precise about node and edge attributes"""
        # Create original graph
        graph1 = create_simple_graph()
        
        # Create similar graph with different attributes
        graph2 = PatternGraph()
        node_a = Node(node_id="A", node_type="A", attributes={"different": "value"})
        node_b = Node(node_id="B", node_type="B")
        edge = Edge(source="A", target="B", edge_type="dependency")
        
        graph2.add_node(node_a)
        graph2.add_node(node_b)
        graph2.add_edge(edge)
        
        # Graphs should NOT be isomorphic due to different attributes
        assert not graph1.is_isomorphic(graph2)
        
        # Create truly identical graph
        graph3 = graph1.copy()
        assert graph1.is_isomorphic(graph3)
    
    def test_invalid_operations(self):
        """Test that invalid operations raise appropriate errors"""
        original_graph = create_simple_graph()
        
        # Test Replicate on non-existent node
        with pytest.raises(ValueError, match="Node C not found"):
            replicate_op = Replicate(node_id='C', count=2)
            replicate_op.apply(original_graph)
        
        # Test CircuitBreaker on non-existent node
        with pytest.raises(ValueError, match="Node C not found"):
            circuit_breaker_op = CircuitBreaker(node_id='C')
            circuit_breaker_op.apply(original_graph)
        
        # Test Cache on non-existent node
        with pytest.raises(ValueError, match="Node C not found"):
            cache_op = Cache(node_id='C')
            cache_op.apply(original_graph)
    
    def test_operation_metadata_preservation(self):
        """Test that operation metadata is properly preserved"""
        original_graph = create_simple_graph()
        
        # Apply operation with specific parameters
        replicate_op = Replicate(node_id='B', count=3, cost_model=self.cost_model)
        replicated_graph = replicate_op.apply(original_graph)
        
        # Verify metadata preservation
        assert replicate_op.applied_cost.metadata['replicated_node'] == 'B'
        assert replicate_op.applied_cost.metadata['count'] == 3
        
        # Test CircuitBreaker metadata
        circuit_breaker_op = CircuitBreaker(
            node_id='B', 
            timeout=15.0, 
            failure_threshold=7,
            cost_model=self.cost_model
        )
        protected_graph = circuit_breaker_op.apply(original_graph)
        
        assert circuit_breaker_op.applied_cost.metadata['protected_node'] == 'B'
        assert circuit_breaker_op.applied_cost.metadata['timeout'] == 15.0
        assert circuit_breaker_op.applied_cost.metadata['failure_threshold'] == 7
    
    def test_complex_graph_roundtrip(self):
        """Test round-trip operations on a more complex graph"""
        # Create more complex graph: A→B→C, A→C
        graph = PatternGraph()
        
        # Add nodes
        for node_id in ['A', 'B', 'C']:
            node = Node(node_id=node_id, node_type=node_id, cost=1.0)
            graph.add_node(node)
        
        # Add edges
        edges = [
            Edge(source='A', target='B', edge_type='dependency'),
            Edge(source='B', target='C', edge_type='dependency'),
            Edge(source='A', target='C', edge_type='shortcut')
        ]
        for edge in edges:
            graph.add_edge(edge)
        
        original_cost = graph.calculate_cost()
        
        # Apply multiple operations
        replicate_op = Replicate(node_id='B', count=2, cost_model=self.cost_model)
        step1_graph = replicate_op.apply(graph)
        
        circuit_breaker_op = CircuitBreaker(node_id='C', cost_model=self.cost_model)
        step2_graph = circuit_breaker_op.apply(step1_graph)
        
        cache_op = Cache(node_id='A', cost_model=self.cost_model)
        step3_graph = cache_op.apply(step2_graph)
        
        # Apply inverses in reverse order
        step4_graph = cache_op.inverse(step3_graph)
        step5_graph = circuit_breaker_op.inverse(step4_graph)
        final_graph = replicate_op.inverse(step5_graph)
        
        # Verify round-trip
        assert final_graph.is_isomorphic(graph)
        assert final_graph.node_count == graph.node_count
        assert final_graph.edge_count == graph.edge_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 