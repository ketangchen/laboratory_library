"""
Enhanced constraint class with static analysis and sampling validation
"""

from typing import Callable, Dict, Any, List, Literal, Optional, Tuple, Union
from deepconstr.train.constr import Constraint
from deepconstr.train.errmsg import ErrorMessage
from deepconstr.static_analysis import SourceAnalyzer, ConstraintMapper, ErrorPatternMatcher
from deepconstr.sampling import ConstraintValidator, ElementSampler
from deepconstr.logger import CONSTR_LOG
import copy


class EnhancedConstraint(Constraint):
    """
    Enhanced constraint class with static analysis and sampling validation
    """
    
    def __init__(self, txt, cot, target, arg_names, dtypes, length=1, 
                 enable_static_analysis=True, enable_sampling=True):
        """
        Initialize enhanced constraint
        
        Args:
            txt: Constraint text
            cot: Chain of thought
            target: Error message target
            arg_names: Argument names
            dtypes: Data types
            length: Constraint length
            enable_static_analysis: Enable static analysis
            enable_sampling: Enable sampling validation
        """
        super().__init__(txt, cot, target, arg_names, dtypes, length)
        
        self.enable_static_analysis = enable_static_analysis
        self.enable_sampling = enable_sampling
        
        # Initialize analysis components
        if self.enable_static_analysis:
            self.source_analyzer = SourceAnalyzer()
            self.constraint_mapper = ConstraintMapper()
            self.error_pattern_matcher = ErrorPatternMatcher()
            self.static_analysis_result = None
        
        if self.enable_sampling:
            self.constraint_validator = ConstraintValidator()
            self.sampling_stats = None
        
        # Enhanced constraint information
        self.enhanced_txt = None
        self.static_analysis_confidence = 0.0
        self.sampling_confidence = 0.0
    
    @staticmethod
    def load(data, enable_static_analysis=True, enable_sampling=True):
        """
        Load enhanced constraint from data
        """
        errmsg = ErrorMessage.load(data["target"])
        dtype_map = errmsg.get_dtypes_map()
        arg_names = list(dtype_map.keys())
        dtypes = list(dtype_map.values())
        
        return EnhancedConstraint(
            data["txt"],
            data["cot"],
            errmsg,
            arg_names,
            dtypes,
            length=data.get("length", 1),
            enable_static_analysis=enable_static_analysis,
            enable_sampling=enable_sampling
        )
    
    def enhance_with_static_analysis(self, operator_name: str, package: str = "torch") -> bool:
        """
        Enhance constraint using static analysis
        
        Args:
            operator_name: Name of the operator
            package: Package name (torch, tensorflow)
            
        Returns:
            True if enhancement was successful
        """
        if not self.enable_static_analysis:
            return False
        
        try:
            # Perform static analysis
            if package == "torch":
                self.static_analysis_result = self.source_analyzer.analyze_torch_operator(operator_name)
            elif package == "tensorflow":
                self.static_analysis_result = self.source_analyzer.analyze_tensorflow_operator(operator_name)
            else:
                CONSTR_LOG.warning(f"Unsupported package: {package}")
                return False
            
            if not self.static_analysis_result:
                return False
            
            # Map error message to constraint
            error_message = self.target.get_core_msg()
            enhanced_mapping = self.constraint_mapper.create_enhanced_error_mapping(
                error_message, operator_name, self.static_analysis_result
            )
            
            if enhanced_mapping:
                self.enhanced_txt = enhanced_mapping.get('enhanced_constraint', self.txt)
                self.static_analysis_confidence = enhanced_mapping.get('confidence', 0.0)
                
                CONSTR_LOG.debug(f"Enhanced constraint: {self.txt} -> {self.enhanced_txt}")
                return True
            
            return False
            
        except Exception as e:
            CONSTR_LOG.warning(f"Static analysis enhancement failed: {e}")
            return False
    
    def validate_with_sampling(self, test_tensors: List[Any], 
                             constraint_func: Optional[Callable] = None) -> bool:
        """
        Validate constraint using sampling
        
        Args:
            test_tensors: List of test tensors
            constraint_func: Optional custom constraint function
            
        Returns:
            True if validation was successful
        """
        if not self.enable_sampling or not test_tensors:
            return True
        
        try:
            # Create constraint function if not provided
            if constraint_func is None:
                constraint_func = self._create_sampling_constraint_function()
            
            # Validate on all test tensors
            validation_results = []
            for tensor in test_tensors:
                is_valid, stats = self.constraint_validator.validate_element_constraint(
                    tensor, self.txt, constraint_func=constraint_func
                )
                validation_results.append((is_valid, stats))
            
            # Calculate overall statistics
            self.sampling_stats = self.constraint_validator.get_validation_statistics(validation_results)
            self.sampling_confidence = self.sampling_stats.get('overall_validity_rate', 0.0)
            
            # Consider validation successful if most tensors pass
            success_rate = self.sampling_stats.get('success_rate', 0.0)
            return success_rate >= 0.8  # 80% success rate threshold
            
        except Exception as e:
            CONSTR_LOG.warning(f"Sampling validation failed: {e}")
            return False
    
    def _create_sampling_constraint_function(self) -> Callable:
        """
        Create constraint function for sampling validation
        """
        # Parse constraint text to create validation function
        constraint_text = self.txt.lower()
        
        if 'element <= ' in constraint_text:
            try:
                max_val = float(constraint_text.split('element <= ')[1].split()[0])
                return lambda element: element <= max_val
            except (IndexError, ValueError):
                CONSTR_LOG.warning(f"Could not parse max value from constraint: {self.txt}")
                return lambda element: True
        
        elif 'element < ' in constraint_text:
            try:
                max_val = float(constraint_text.split('element < ')[1].split()[0])
                return lambda element: element < max_val
            except (IndexError, ValueError):
                CONSTR_LOG.warning(f"Could not parse max value from constraint: {self.txt}")
                return lambda element: True
        
        elif 'all(' in constraint_text and 'element' in constraint_text:
            # Handle "all(element <= n for element in tensor)" format
            try:
                # Extract the condition from the constraint
                if 'element <= ' in constraint_text:
                    max_val = float(constraint_text.split('element <= ')[1].split()[0])
                    return lambda element: element <= max_val
                elif 'element < ' in constraint_text:
                    max_val = float(constraint_text.split('element < ')[1].split()[0])
                    return lambda element: element < max_val
            except (IndexError, ValueError):
                CONSTR_LOG.warning(f"Could not parse constraint: {self.txt}")
                return lambda element: True
        
        else:
            # Default to always valid for non-element constraints
            return lambda element: True
    
    def get_enhanced_constraint_text(self) -> str:
        """
        Get enhanced constraint text
        """
        if self.enhanced_txt:
            return self.enhanced_txt
        return self.txt
    
    def get_confidence_score(self) -> float:
        """
        Get overall confidence score combining static analysis and sampling
        """
        if self.enable_static_analysis and self.enable_sampling:
            # Weighted average of both confidence scores
            return (self.static_analysis_confidence * 0.6 + self.sampling_confidence * 0.4)
        elif self.enable_static_analysis:
            return self.static_analysis_confidence
        elif self.enable_sampling:
            return self.sampling_confidence
        else:
            return 1.0  # Default confidence if no enhancement
    
    def get_enhancement_info(self) -> Dict[str, Any]:
        """
        Get information about constraint enhancements
        """
        info = {
            'original_constraint': self.txt,
            'enhanced_constraint': self.get_enhanced_constraint_text(),
            'static_analysis_enabled': self.enable_static_analysis,
            'sampling_enabled': self.enable_sampling,
            'static_analysis_confidence': self.static_analysis_confidence,
            'sampling_confidence': self.sampling_confidence,
            'overall_confidence': self.get_confidence_score()
        }
        
        if self.static_analysis_result:
            info['static_analysis_result'] = {
                'constraints_found': len(self.static_analysis_result.get('constraints', [])),
                'error_mappings_found': len(self.static_analysis_result.get('error_mappings', {}))
            }
        
        if self.sampling_stats:
            info['sampling_stats'] = self.sampling_stats
        
        return info
    
    def dump(self) -> Dict[str, Any]:
        """
        Dump enhanced constraint data
        """
        base_data = super().dump()
        
        # Add enhancement information
        base_data.update({
            'enhanced_txt': self.get_enhanced_constraint_text(),
            'static_analysis_confidence': self.static_analysis_confidence,
            'sampling_confidence': self.sampling_confidence,
            'overall_confidence': self.get_confidence_score(),
            'enhancement_info': self.get_enhancement_info()
        })
        
        return base_data
    
    def is_enhanced(self) -> bool:
        """
        Check if constraint has been enhanced
        """
        return (self.enhanced_txt is not None or 
                self.static_analysis_confidence > 0 or 
                self.sampling_confidence > 0)
    
    def apply_enhancements(self, operator_name: str, package: str = "torch", 
                          test_tensors: Optional[List[Any]] = None) -> bool:
        """
        Apply all enhancements to the constraint
        
        Args:
            operator_name: Name of the operator
            package: Package name
            test_tensors: Optional test tensors for sampling validation
            
        Returns:
            True if enhancements were applied successfully
        """
        success = True
        
        # Apply static analysis enhancement
        if self.enable_static_analysis:
            static_success = self.enhance_with_static_analysis(operator_name, package)
            if not static_success:
                CONSTR_LOG.debug(f"Static analysis enhancement failed for {operator_name}")
            success = success and static_success
        
        # Apply sampling validation
        if self.enable_sampling and test_tensors:
            sampling_success = self.validate_with_sampling(test_tensors)
            if not sampling_success:
                CONSTR_LOG.debug(f"Sampling validation failed for {operator_name}")
            success = success and sampling_success
        
        return success
