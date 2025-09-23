"""
Constraint validator for element-level constraints using sampling
"""

import numpy as np
from typing import List, Tuple, Optional, Any, Dict, Callable
from deepconstr.logger import CONSTR_LOG


class ConstraintValidator:
    """
    Validates element-level constraints using sampling-based approach
    """
    
    def __init__(self, sampler=None):
        """
        Initialize constraint validator
        
        Args:
            sampler: ElementSampler instance for sampling elements
        """
        from .element_sampler import ElementSampler
        self.sampler = sampler or ElementSampler()
    
    def validate_element_constraint(self, tensor: Any, constraint: str, 
                                  max_value: Optional[float] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate element-level constraint using sampling
        
        Args:
            tensor: The tensor to validate
            constraint: Constraint string (e.g., "element <= n")
            max_value: Maximum allowed value for elements
            
        Returns:
            Tuple of (is_valid, statistics)
        """
        try:
            # Parse constraint to create validation function
            constraint_func = self._create_constraint_function(constraint, max_value)
            
            # Use sampler to validate constraint
            is_valid, stats = self.sampler.sample_tensor_elements(tensor, constraint_func)
            
            # Add constraint information to stats
            stats['constraint'] = constraint
            stats['max_value'] = max_value
            
            return is_valid, stats
            
        except Exception as e:
            CONSTR_LOG.warning(f"Constraint validation failed: {e}")
            return False, {'error': str(e), 'constraint': constraint}
    
    def validate_element_constraint_stratified(self, tensor: Any, constraint: str,
                                            max_value: Optional[float] = None,
                                            strata_count: int = 5) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate element-level constraint using stratified sampling
        
        Args:
            tensor: The tensor to validate
            constraint: Constraint string
            max_value: Maximum allowed value for elements
            strata_count: Number of strata for stratification
            
        Returns:
            Tuple of (is_valid, statistics)
        """
        try:
            constraint_func = self._create_constraint_function(constraint, max_value)
            
            is_valid, stats = self.sampler.sample_with_stratification(
                tensor, constraint_func, strata_count
            )
            
            stats['constraint'] = constraint
            stats['max_value'] = max_value
            stats['strata_count'] = strata_count
            
            return is_valid, stats
            
        except Exception as e:
            CONSTR_LOG.warning(f"Stratified constraint validation failed: {e}")
            return False, {'error': str(e), 'constraint': constraint}
    
    def validate_element_constraint_adaptive(self, tensor: Any, constraint: str,
                                           max_value: Optional[float] = None,
                                           initial_sample_rate: float = 0.05) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate element-level constraint using adaptive sampling
        
        Args:
            tensor: The tensor to validate
            constraint: Constraint string
            max_value: Maximum allowed value for elements
            initial_sample_rate: Initial sampling rate
            
        Returns:
            Tuple of (is_valid, statistics)
        """
        try:
            constraint_func = self._create_constraint_function(constraint, max_value)
            
            is_valid, stats = self.sampler.adaptive_sampling(
                tensor, constraint_func, initial_sample_rate
            )
            
            stats['constraint'] = constraint
            stats['max_value'] = max_value
            stats['initial_sample_rate'] = initial_sample_rate
            
            return is_valid, stats
            
        except Exception as e:
            CONSTR_LOG.warning(f"Adaptive constraint validation failed: {e}")
            return False, {'error': str(e), 'constraint': constraint}
    
    def _create_constraint_function(self, constraint: str, max_value: Optional[float] = None) -> Callable:
        """
        Create constraint validation function from constraint string
        
        Args:
            constraint: Constraint string
            max_value: Maximum allowed value
            
        Returns:
            Function that validates a single element
        """
        if max_value is not None:
            # Use provided max_value
            return lambda element: element <= max_value
        
        # Parse constraint string
        constraint_lower = constraint.lower().strip()
        
        if 'element <= ' in constraint_lower:
            # Extract max value from constraint
            try:
                max_val = float(constraint_lower.split('element <= ')[1].split()[0])
                return lambda element: element <= max_val
            except (IndexError, ValueError):
                CONSTR_LOG.warning(f"Could not parse max value from constraint: {constraint}")
                return lambda element: True  # Default to always valid
        
        elif 'element < ' in constraint_lower:
            try:
                max_val = float(constraint_lower.split('element < ')[1].split()[0])
                return lambda element: element < max_val
            except (IndexError, ValueError):
                CONSTR_LOG.warning(f"Could not parse max value from constraint: {constraint}")
                return lambda element: True
        
        elif 'element >= ' in constraint_lower:
            try:
                min_val = float(constraint_lower.split('element >= ')[1].split()[0])
                return lambda element: element >= min_val
            except (IndexError, ValueError):
                CONSTR_LOG.warning(f"Could not parse min value from constraint: {constraint}")
                return lambda element: True
        
        elif 'element > ' in constraint_lower:
            try:
                min_val = float(constraint_lower.split('element > ')[1].split()[0])
                return lambda element: element > min_val
            except (IndexError, ValueError):
                CONSTR_LOG.warning(f"Could not parse min value from constraint: {constraint}")
                return lambda element: True
        
        else:
            CONSTR_LOG.warning(f"Unsupported constraint format: {constraint}")
            return lambda element: True  # Default to always valid
    
    def batch_validate_constraints(self, tensors: List[Any], constraints: List[str],
                                 max_values: Optional[List[float]] = None) -> List[Tuple[bool, Dict[str, Any]]]:
        """
        Validate multiple constraints on multiple tensors
        
        Args:
            tensors: List of tensors to validate
            constraints: List of constraint strings
            max_values: Optional list of max values for each constraint
            
        Returns:
            List of validation results
        """
        if max_values is None:
            max_values = [None] * len(constraints)
        
        if len(tensors) != len(constraints):
            raise ValueError("Number of tensors must match number of constraints")
        
        results = []
        for tensor, constraint, max_value in zip(tensors, constraints, max_values):
            try:
                result = self.validate_element_constraint(tensor, constraint, max_value)
                results.append(result)
            except Exception as e:
                CONSTR_LOG.warning(f"Batch validation failed for constraint {constraint}: {e}")
                results.append((False, {'error': str(e), 'constraint': constraint}))
        
        return results
    
    def get_validation_statistics(self, validation_results: List[Tuple[bool, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Get statistics from batch validation results
        
        Args:
            validation_results: Results from batch validation
            
        Returns:
            Statistics dictionary
        """
        total_validations = len(validation_results)
        successful_validations = sum(1 for is_valid, _ in validation_results if is_valid)
        
        # Calculate average statistics
        total_elements = 0
        total_sampled = 0
        total_valid = 0
        total_invalid = 0
        
        for is_valid, stats in validation_results:
            if 'error' not in stats:
                total_elements += stats.get('total_elements', 0)
                total_sampled += stats.get('sampled_elements', 0)
                total_valid += stats.get('valid_elements', 0)
                total_invalid += stats.get('invalid_elements', 0)
        
        # Calculate rates
        success_rate = successful_validations / total_validations if total_validations > 0 else 0
        overall_validity_rate = total_valid / total_sampled if total_sampled > 0 else 0
        average_sample_rate = total_sampled / total_elements if total_elements > 0 else 0
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'success_rate': success_rate,
            'total_elements_processed': total_elements,
            'total_elements_sampled': total_sampled,
            'overall_validity_rate': overall_validity_rate,
            'average_sample_rate': average_sample_rate,
            'total_invalid_elements': total_invalid
        }
