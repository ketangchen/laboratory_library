"""
Constraint mapper for mapping error messages to actual constraints
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from deepconstr.logger import CONSTR_LOG


class ConstraintMapper:
    """
    Maps error messages to actual constraints using static analysis
    """
    
    def __init__(self):
        self.error_constraint_mappings = {}
        self.constraint_templates = {
            'kernel_size_too_large': 'kernel_size <= input_size + 2 * padding',
            'negative_dimensions': 'all(dim >= 0 for dim in shape)',
            'shape_mismatch': 'shape1 == shape2',
            'dimension_mismatch': 'len(shape1) == len(shape2)',
            'size_too_large': 'size <= max_size',
            'element_too_large': 'all(element <= max_value for element in tensor)'
        }
    
    def map_error_to_constraint(self, error_message: str, operator_name: str, 
                               static_analysis_result: Dict[str, Any]) -> Optional[str]:
        """
        Map error message to actual constraint using static analysis
        
        Args:
            error_message: The error message to map
            operator_name: Name of the operator
            static_analysis_result: Result from static analysis
            
        Returns:
            Mapped constraint string or None
        """
        # First try exact match from static analysis
        if 'error_mappings' in static_analysis_result:
            for msg, constraints in static_analysis_result['error_mappings'].items():
                if self._messages_similar(error_message, msg):
                    return constraints[0] if constraints else None
        
        # Try pattern-based mapping
        constraint = self._pattern_based_mapping(error_message, operator_name)
        if constraint:
            return constraint
        
        # Try template-based mapping
        constraint = self._template_based_mapping(error_message, operator_name)
        if constraint:
            return constraint
        
        CONSTR_LOG.debug(f"No constraint mapping found for error: {error_message}")
        return None
    
    def _messages_similar(self, msg1: str, msg2: str, threshold: float = 0.7) -> bool:
        """
        Check if two error messages are similar
        """
        # Simple similarity based on common words
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity >= threshold
    
    def _pattern_based_mapping(self, error_message: str, operator_name: str) -> Optional[str]:
        """
        Map error message to constraint using pattern matching
        """
        error_lower = error_message.lower()
        
        # Kernel size related errors
        if 'kernel' in error_lower and ('large' in error_lower or 'big' in error_lower):
            return 'kernel_size <= input_size + 2 * padding'
        
        # Shape related errors
        if 'shape' in error_lower and ('mismatch' in error_lower or 'incompatible' in error_lower):
            return 'shape1 == shape2'
        
        # Dimension related errors
        if 'dimension' in error_lower and ('negative' in error_lower or 'invalid' in error_lower):
            return 'all(dim >= 0 for dim in shape)'
        
        # Size related errors
        if 'size' in error_lower and ('large' in error_lower or 'exceed' in error_lower):
            return 'size <= max_size'
        
        # Element related errors
        if 'element' in error_lower and ('large' in error_lower or 'exceed' in error_lower):
            return 'all(element <= max_value for element in tensor)'
        
        return None
    
    def _template_based_mapping(self, error_message: str, operator_name: str) -> Optional[str]:
        """
        Map error message to constraint using predefined templates
        """
        error_lower = error_message.lower()
        
        # Check against known templates
        for template_key, template_constraint in self.constraint_templates.items():
            if self._matches_template(error_lower, template_key):
                return template_constraint
        
        return None
    
    def _matches_template(self, error_message: str, template_key: str) -> bool:
        """
        Check if error message matches a template
        """
        template_patterns = {
            'kernel_size_too_large': ['kernel', 'size', 'large', 'big'],
            'negative_dimensions': ['negative', 'dimension'],
            'shape_mismatch': ['shape', 'mismatch', 'incompatible'],
            'dimension_mismatch': ['dimension', 'mismatch'],
            'size_too_large': ['size', 'large', 'exceed'],
            'element_too_large': ['element', 'large', 'exceed']
        }
        
        if template_key not in template_patterns:
            return False
        
        required_words = template_patterns[template_key]
        return all(word in error_message for word in required_words)
    
    def enhance_constraint_with_static_analysis(self, constraint: str, 
                                             static_analysis_result: Dict[str, Any]) -> str:
        """
        Enhance constraint with information from static analysis
        """
        if 'constraints' in static_analysis_result:
            # Find related constraints from static analysis
            related_constraints = self._find_related_constraints(
                constraint, static_analysis_result['constraints']
            )
            
            if related_constraints:
                # Combine with original constraint
                return f"({constraint}) and ({' and '.join(related_constraints)})"
        
        return constraint
    
    def _find_related_constraints(self, constraint: str, 
                                static_constraints: List[Dict[str, Any]]) -> List[str]:
        """
        Find constraints related to the given constraint from static analysis
        """
        related = []
        constraint_lower = constraint.lower()
        
        for static_constraint in static_constraints:
            if static_constraint.get('type') == 'comparison':
                left = static_constraint.get('left', '').lower()
                right = static_constraint.get('right', '').lower()
                op = static_constraint.get('operator', '')
                
                # Check if this constraint is related
                if (any(keyword in left for keyword in ['kernel', 'size', 'shape', 'dim']) or
                    any(keyword in right for keyword in ['kernel', 'size', 'shape', 'dim'])):
                    
                    constraint_text = f"{left} {op} {right}"
                    related.append(constraint_text)
        
        return related
    
    def create_enhanced_error_mapping(self, error_message: str, operator_name: str,
                                     static_analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create enhanced error mapping with static analysis information
        """
        base_constraint = self.map_error_to_constraint(
            error_message, operator_name, static_analysis_result
        )
        
        if not base_constraint:
            return {}
        
        enhanced_constraint = self.enhance_constraint_with_static_analysis(
            base_constraint, static_analysis_result
        )
        
        return {
            'original_error': error_message,
            'operator': operator_name,
            'base_constraint': base_constraint,
            'enhanced_constraint': enhanced_constraint,
            'static_analysis_used': bool(static_analysis_result),
            'confidence': self._calculate_confidence(error_message, base_constraint)
        }
    
    def _calculate_confidence(self, error_message: str, constraint: str) -> float:
        """
        Calculate confidence score for the mapping
        """
        # Simple confidence based on keyword overlap
        error_words = set(error_message.lower().split())
        constraint_words = set(constraint.lower().split())
        
        if not error_words or not constraint_words:
            return 0.0
        
        intersection = error_words.intersection(constraint_words)
        return len(intersection) / max(len(error_words), len(constraint_words))
