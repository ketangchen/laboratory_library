"""
Error pattern matcher for identifying and categorizing error messages
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from deepconstr.logger import CONSTR_LOG


class ErrorPatternMatcher:
    """
    Matches error messages to patterns and extracts constraint information
    """
    
    def __init__(self):
        self.error_patterns = {
            'kernel_size_error': {
                'pattern': r'kernel.*size.*(?:too large|exceeds|greater than)',
                'constraint_type': 'kernel_size_constraint',
                'extract_vars': ['kernel_size', 'input_size', 'padding']
            },
            'shape_error': {
                'pattern': r'shape.*(?:mismatch|incompatible|invalid)',
                'constraint_type': 'shape_constraint',
                'extract_vars': ['shape1', 'shape2']
            },
            'dimension_error': {
                'pattern': r'dimension.*(?:negative|invalid|out of range)',
                'constraint_type': 'dimension_constraint',
                'extract_vars': ['dimension', 'shape']
            },
            'size_error': {
                'pattern': r'size.*(?:too large|exceeds|greater than)',
                'constraint_type': 'size_constraint',
                'extract_vars': ['size', 'max_size']
            },
            'element_error': {
                'pattern': r'element.*(?:too large|exceeds|greater than)',
                'constraint_type': 'element_constraint',
                'extract_vars': ['element', 'max_value']
            }
        }
        
        self.constraint_generators = {
            'kernel_size_constraint': self._generate_kernel_size_constraint,
            'shape_constraint': self._generate_shape_constraint,
            'dimension_constraint': self._generate_dimension_constraint,
            'size_constraint': self._generate_size_constraint,
            'element_constraint': self._generate_element_constraint
        }
    
    def match_error_pattern(self, error_message: str) -> Optional[Dict[str, Any]]:
        """
        Match error message to known patterns
        
        Args:
            error_message: The error message to analyze
            
        Returns:
            Dict containing pattern match information or None
        """
        error_lower = error_message.lower()
        
        for pattern_name, pattern_info in self.error_patterns.items():
            if re.search(pattern_info['pattern'], error_lower, re.IGNORECASE):
                return {
                    'pattern_name': pattern_name,
                    'constraint_type': pattern_info['constraint_type'],
                    'extract_vars': pattern_info['extract_vars'],
                    'matched_text': re.search(pattern_info['pattern'], error_lower, re.IGNORECASE).group()
                }
        
        return None
    
    def extract_constraint_from_error(self, error_message: str, 
                                    pattern_match: Dict[str, Any]) -> Optional[str]:
        """
        Extract constraint from error message using pattern match
        
        Args:
            error_message: The error message
            pattern_match: Pattern match information
            
        Returns:
            Generated constraint string or None
        """
        constraint_type = pattern_match['constraint_type']
        
        if constraint_type in self.constraint_generators:
            return self.constraint_generators[constraint_type](error_message, pattern_match)
        
        return None
    
    def _generate_kernel_size_constraint(self, error_message: str, 
                                       pattern_match: Dict[str, Any]) -> str:
        """
        Generate kernel size constraint from error message
        """
        # Extract numerical values from error message
        numbers = re.findall(r'\d+', error_message)
        
        if len(numbers) >= 2:
            kernel_size, input_size = numbers[0], numbers[1]
            return f"kernel_size <= {input_size} + 2 * padding"
        else:
            return "kernel_size <= input_size + 2 * padding"
    
    def _generate_shape_constraint(self, error_message: str, 
                                 pattern_match: Dict[str, Any]) -> str:
        """
        Generate shape constraint from error message
        """
        # Look for shape information in error message
        shape_pattern = r'\[([^\]]+)\]'
        shapes = re.findall(shape_pattern, error_message)
        
        if len(shapes) >= 2:
            return f"shape1 == shape2  # {shapes[0]} vs {shapes[1]}"
        else:
            return "shape1 == shape2"
    
    def _generate_dimension_constraint(self, error_message: str, 
                                       pattern_match: Dict[str, Any]) -> str:
        """
        Generate dimension constraint from error message
        """
        # Extract dimension information
        dim_pattern = r'dimension\s*(\d+)'
        dim_match = re.search(dim_pattern, error_message, re.IGNORECASE)
        
        if dim_match:
            dim = dim_match.group(1)
            return f"all(dim >= 0 for dim in shape) and len(shape) == {dim}"
        else:
            return "all(dim >= 0 for dim in shape)"
    
    def _generate_size_constraint(self, error_message: str, 
                                pattern_match: Dict[str, Any]) -> str:
        """
        Generate size constraint from error message
        """
        # Extract size information
        size_pattern = r'size\s*(\d+)'
        size_match = re.search(size_pattern, error_message, re.IGNORECASE)
        
        if size_match:
            size = size_match.group(1)
            return f"size <= {size}"
        else:
            return "size <= max_size"
    
    def _generate_element_constraint(self, error_message: str, 
                                   pattern_match: Dict[str, Any]) -> str:
        """
        Generate element constraint from error message
        """
        # Extract element value information
        value_pattern = r'(\d+(?:\.\d+)?)'
        values = re.findall(value_pattern, error_message)
        
        if values:
            max_value = values[-1]  # Use the last number found
            return f"all(element <= {max_value} for element in tensor)"
        else:
            return "all(element <= max_value for element in tensor)"
    
    def analyze_error_message(self, error_message: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of error message
        
        Args:
            error_message: The error message to analyze
            
        Returns:
            Dict containing analysis results
        """
        pattern_match = self.match_error_pattern(error_message)
        
        if not pattern_match:
            return {
                'matched': False,
                'error_message': error_message,
                'constraint': None
            }
        
        constraint = self.extract_constraint_from_error(error_message, pattern_match)
        
        return {
            'matched': True,
            'error_message': error_message,
            'pattern_match': pattern_match,
            'constraint': constraint,
            'confidence': self._calculate_pattern_confidence(error_message, pattern_match)
        }
    
    def _calculate_pattern_confidence(self, error_message: str, 
                                    pattern_match: Dict[str, Any]) -> float:
        """
        Calculate confidence score for pattern match
        """
        # Simple confidence based on pattern match quality
        matched_text = pattern_match.get('matched_text', '')
        
        if not matched_text:
            return 0.0
        
        # Calculate overlap ratio
        error_words = set(error_message.lower().split())
        matched_words = set(matched_text.lower().split())
        
        if not error_words or not matched_words:
            return 0.0
        
        intersection = error_words.intersection(matched_words)
        return len(intersection) / len(matched_words)
    
    def batch_analyze_errors(self, error_messages: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple error messages in batch
        
        Args:
            error_messages: List of error messages to analyze
            
        Returns:
            List of analysis results
        """
        results = []
        
        for error_message in error_messages:
            try:
                result = self.analyze_error_message(error_message)
                results.append(result)
            except Exception as e:
                CONSTR_LOG.warning(f"Failed to analyze error message: {error_message}, {e}")
                results.append({
                    'matched': False,
                    'error_message': error_message,
                    'constraint': None,
                    'error': str(e)
                })
        
        return results
    
    def get_constraint_statistics(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics from batch analysis results
        
        Args:
            analysis_results: Results from batch analysis
            
        Returns:
            Statistics dictionary
        """
        total = len(analysis_results)
        matched = sum(1 for result in analysis_results if result.get('matched', False))
        
        constraint_types = {}
        for result in analysis_results:
            if result.get('matched', False):
                pattern_match = result.get('pattern_match', {})
                constraint_type = pattern_match.get('constraint_type', 'unknown')
                constraint_types[constraint_type] = constraint_types.get(constraint_type, 0) + 1
        
        return {
            'total_errors': total,
            'matched_errors': matched,
            'match_rate': matched / total if total > 0 else 0,
            'constraint_type_distribution': constraint_types,
            'average_confidence': sum(
                result.get('confidence', 0) for result in analysis_results
            ) / total if total > 0 else 0
        }
