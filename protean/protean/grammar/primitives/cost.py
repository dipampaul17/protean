"""
Cost Tracking for Pattern Operations
Tracks resource costs and ensures cost-awareness in transformations
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import math


class CostType(Enum):
    """Types of costs in pattern operations"""
    COMPUTATIONAL = "computational"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    OPERATIONAL = "operational"


@dataclass
class OperationCost:
    """Cost tracking for a single operation"""
    op_type: str
    cost_delta: float
    cost_breakdown: Dict[CostType, float]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        # Validate that only Cache operations and inverse operations can have negative cost
        if self.cost_delta < 0 and self.op_type != 'Cache' and not self.op_type.startswith('Inverse'):
            raise ValueError(f"Operation {self.op_type} cannot have negative cost delta: {self.cost_delta}")
    
    @property 
    def is_cost_reducing(self) -> bool:
        """Check if operation reduces overall cost"""
        return self.cost_delta < 0
    
    @property
    def is_cache_operation(self) -> bool:
        """Check if this is a cache operation (allowed to reduce cost)"""
        return self.op_type == 'Cache'
    
    def validate(self) -> bool:
        """Validate cost constraints"""
        # Only Cache operations and inverse operations can have negative cost
        if self.cost_delta < 0 and not self.is_cache_operation and not self.op_type.startswith('Inverse'):
            return False
        return True


class CostModel:
    """
    Cost model for pattern graph operations
    Tracks costs and ensures operations follow cost constraints
    """
    
    def __init__(self):
        self.base_costs = {
            CostType.COMPUTATIONAL: 1.0,
            CostType.MEMORY: 0.5,
            CostType.NETWORK: 2.0,
            CostType.STORAGE: 0.3,
            CostType.OPERATIONAL: 1.5
        }
        self.operation_history: List[OperationCost] = []
    
    def calculate_operation_cost(self, op_type: str, 
                                 node_count_delta: int = 0,
                                 edge_count_delta: int = 0,
                                 complexity_factor: float = 1.0,
                                 metadata: Optional[Dict[str, Any]] = None) -> OperationCost:
        """Calculate cost for a graph operation"""
        
        cost_breakdown = {cost_type: 0.0 for cost_type in CostType}
        
        # Base operation costs
        if op_type == 'Replicate':
            cost_breakdown[CostType.COMPUTATIONAL] = abs(node_count_delta) * 2.0 * complexity_factor
            cost_breakdown[CostType.MEMORY] = abs(node_count_delta) * 1.0 * complexity_factor
            cost_breakdown[CostType.OPERATIONAL] = 1.0
            
        elif op_type == 'CircuitBreaker':
            cost_breakdown[CostType.COMPUTATIONAL] = 0.5 * complexity_factor
            cost_breakdown[CostType.NETWORK] = 1.0 * complexity_factor
            cost_breakdown[CostType.OPERATIONAL] = 2.0
            
        elif op_type == 'Cache':
            # Cache can reduce costs (negative cost delta allowed)
            cost_breakdown[CostType.COMPUTATIONAL] = -1.0 * complexity_factor  # Saves compute
            cost_breakdown[CostType.MEMORY] = 2.0 * complexity_factor  # Uses memory
            cost_breakdown[CostType.NETWORK] = -0.5 * complexity_factor  # Reduces network calls
            
        elif op_type == 'Split':
            cost_breakdown[CostType.COMPUTATIONAL] = 1.5 * complexity_factor
            cost_breakdown[CostType.OPERATIONAL] = 1.0
            
        elif op_type == 'Merge':
            cost_breakdown[CostType.COMPUTATIONAL] = 2.0 * complexity_factor
            cost_breakdown[CostType.OPERATIONAL] = 1.5
            
        else:
            # Default operation cost
            cost_breakdown[CostType.COMPUTATIONAL] = 1.0 * complexity_factor
            cost_breakdown[CostType.OPERATIONAL] = 0.5
        
        # Add edge costs
        cost_breakdown[CostType.NETWORK] += abs(edge_count_delta) * 0.5
        
        # Calculate total cost delta
        cost_delta = sum(
            cost * self.base_costs[cost_type] 
            for cost_type, cost in cost_breakdown.items()
        )
        
        operation_cost = OperationCost(
            op_type=op_type,
            cost_delta=cost_delta,
            cost_breakdown=cost_breakdown,
            metadata=metadata or {}
        )
        
        # Validate cost constraints
        if not operation_cost.validate():
            raise ValueError(f"Invalid cost for operation {op_type}: {cost_delta}")
        
        return operation_cost
    
    def record_operation(self, operation_cost: OperationCost) -> None:
        """Record an operation cost in history"""
        self.operation_history.append(operation_cost)
    
    def get_total_cost(self) -> float:
        """Get total cost of all operations"""
        return sum(op.cost_delta for op in self.operation_history)
    
    def get_cost_by_type(self, cost_type: CostType) -> float:
        """Get total cost by type"""
        return sum(
            op.cost_breakdown.get(cost_type, 0.0) 
            for op in self.operation_history
        )
    
    def validate_cost_constraints(self) -> bool:
        """Validate all operations follow cost constraints"""
        for op in self.operation_history:
            if not op.validate():
                return False
        return True
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of all costs"""
        return {
            'total_cost': self.get_total_cost(),
            'operation_count': len(self.operation_history),
            'cost_by_type': {
                cost_type.value: self.get_cost_by_type(cost_type)
                for cost_type in CostType
            },
            'operations': [
                {
                    'op_type': op.op_type,
                    'cost_delta': op.cost_delta,
                    'is_cost_reducing': op.is_cost_reducing
                }
                for op in self.operation_history
            ]
        }
    
    def reset(self) -> None:
        """Reset operation history"""
        self.operation_history.clear() 