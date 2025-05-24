"""
Core Graph Structures for Pattern Representation
Built on NetworkX for efficient graph operations and analysis
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import uuid
import networkx as nx
import json
from copy import deepcopy


@dataclass
class Node:
    """A node in the pattern graph"""
    node_id: str
    node_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'attributes': self.attributes,
            'cost': self.cost
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        return cls(**data)


@dataclass 
class Edge:
    """An edge in the pattern graph"""
    source: str
    target: str
    edge_type: str = "default"
    attributes: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'attributes': self.attributes,
            'weight': self.weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Edge':
        return cls(**data)


class PatternGraph:
    """
    A pattern graph representing infrastructure relationships
    Built on NetworkX DiGraph for efficient operations
    """
    
    def __init__(self, graph_id: Optional[str] = None):
        self.graph_id = graph_id or str(uuid.uuid4())[:8]
        self.graph = nx.DiGraph()
        self._metadata: Dict[str, Any] = {}
        self._cost_cache: Optional[float] = None
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata
    
    @metadata.setter 
    def metadata(self, value: Dict[str, Any]):
        self._metadata = value
        self._cost_cache = None  # Invalidate cost cache
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        self.graph.add_node(node.node_id, **node.to_dict())
        self._cost_cache = None
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph"""
        self.graph.add_edge(edge.source, edge.target, **edge.to_dict())
        self._cost_cache = None
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges"""
        if node_id in self.graph:
            self.graph.remove_node(node_id)
            self._cost_cache = None
    
    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge"""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            self._cost_cache = None
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID"""
        if node_id not in self.graph:
            return None
        
        data = self.graph.nodes[node_id]
        return Node.from_dict(data)
    
    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        """Get an edge by source and target"""
        if not self.graph.has_edge(source, target):
            return None
        
        data = self.graph.edges[source, target]
        return Edge.from_dict(data)
    
    def get_nodes(self) -> List[Node]:
        """Get all nodes"""
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append(Node.from_dict(data))
        return nodes
    
    def get_edges(self) -> List[Edge]:
        """Get all edges"""
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append(Edge.from_dict(data))
        return edges
    
    def get_successors(self, node_id: str) -> List[str]:
        """Get successor node IDs"""
        return list(self.graph.successors(node_id))
    
    def get_predecessors(self, node_id: str) -> List[str]:
        """Get predecessor node IDs"""
        return list(self.graph.predecessors(node_id))
    
    def has_node(self, node_id: str) -> bool:
        """Check if node exists"""
        return node_id in self.graph
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if edge exists"""
        return self.graph.has_edge(source, target)
    
    @property
    def node_count(self) -> int:
        """Number of nodes"""
        return len(self.graph.nodes)
    
    @property 
    def edge_count(self) -> int:
        """Number of edges"""
        return len(self.graph.edges)
    
    def calculate_cost(self) -> float:
        """Calculate total graph cost"""
        if self._cost_cache is not None:
            return self._cost_cache
        
        total_cost = 0.0
        
        # Node costs
        for _, data in self.graph.nodes(data=True):
            total_cost += data.get('cost', 1.0)
        
        # Edge costs  
        for _, _, data in self.graph.edges(data=True):
            total_cost += data.get('weight', 1.0)
        
        # Metadata costs
        total_cost += self._metadata.get('base_cost', 0.0)
        
        self._cost_cache = total_cost
        return total_cost
    
    def copy(self) -> 'PatternGraph':
        """Create a deep copy of the graph"""
        new_graph = PatternGraph(graph_id=f"{self.graph_id}_copy")
        new_graph.graph = self.graph.copy()
        new_graph._metadata = deepcopy(self._metadata)
        return new_graph
    
    def is_isomorphic(self, other: 'PatternGraph') -> bool:
        """Check if graphs are isomorphic including node/edge attributes"""
        def node_match(n1, n2):
            # Compare all node attributes except node_id
            attrs1 = {k: v for k, v in n1.items() if k != 'node_id'}
            attrs2 = {k: v for k, v in n2.items() if k != 'node_id'}
            return attrs1 == attrs2
        
        def edge_match(e1, e2):
            # Compare all edge attributes except source/target
            attrs1 = {k: v for k, v in e1.items() if k not in ['source', 'target']}
            attrs2 = {k: v for k, v in e2.items() if k not in ['source', 'target']}
            return attrs1 == attrs2
        
        return nx.is_isomorphic(
            self.graph, other.graph,
            node_match=node_match,
            edge_match=edge_match
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary"""
        return {
            'graph_id': self.graph_id,
            'nodes': [node.to_dict() for node in self.get_nodes()],
            'edges': [edge.to_dict() for edge in self.get_edges()],
            'metadata': self._metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternGraph':
        """Deserialize graph from dictionary"""
        graph = cls(graph_id=data['graph_id'])
        graph._metadata = data.get('metadata', {})
        
        # Add nodes
        for node_data in data['nodes']:
            node = Node.from_dict(node_data)
            graph.add_node(node)
        
        # Add edges
        for edge_data in data['edges']:
            edge = Edge.from_dict(edge_data)
            graph.add_edge(edge)
        
        return graph
    
    def to_networkx(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph"""
        return self.graph.copy()
    
    @classmethod
    def from_networkx(cls, nx_graph: nx.DiGraph, graph_id: Optional[str] = None) -> 'PatternGraph':
        """Create PatternGraph from NetworkX DiGraph"""
        graph = cls(graph_id=graph_id)
        
        # Add nodes
        for node_id, data in nx_graph.nodes(data=True):
            node_data = data.copy()
            if 'node_id' not in node_data:
                node_data['node_id'] = node_id
            if 'node_type' not in node_data:
                node_data['node_type'] = 'unknown'
            
            node = Node.from_dict(node_data)
            graph.add_node(node)
        
        # Add edges
        for source, target, data in nx_graph.edges(data=True):
            edge_data = data.copy()
            edge_data['source'] = source
            edge_data['target'] = target
            
            edge = Edge.from_dict(edge_data)
            graph.add_edge(edge)
        
        return graph
    
    def __str__(self) -> str:
        return f"PatternGraph(id={self.graph_id}, nodes={self.node_count}, edges={self.edge_count})"
    
    def __repr__(self) -> str:
        return self.__str__()


def create_simple_graph(node_a_type: str = "A", node_b_type: str = "B") -> PatternGraph:
    """Create a simple A→B pattern graph for testing"""
    graph = PatternGraph()
    
    # Add nodes
    node_a = Node(node_id="A", node_type=node_a_type, cost=1.0)
    node_b = Node(node_id="B", node_type=node_b_type, cost=1.0)
    
    graph.add_node(node_a)
    graph.add_node(node_b)
    
    # Add edge A→B
    edge = Edge(source="A", target="B", edge_type="dependency", weight=1.0)
    graph.add_edge(edge)
    
    return graph 