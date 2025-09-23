"""
Enhanced training loop with static analysis and sampling validation
"""

import copy
from typing import Dict, Any, List, Tuple, Optional
from omegaconf import DictConfig
from deepconstr.train.run import TrainingLoop
from deepconstr.train.constr import Constraint
from deepconstr.train.enhanced_constr import EnhancedConstraint
from deepconstr.train.errmsg import ErrorMessage
from deepconstr.logger import TRAIN_LOG


class EnhancedTrainingLoop(TrainingLoop):
    """
    Enhanced training loop with static analysis and sampling validation
    """
    
    def __init__(self, cfg: DictConfig, enable_static_analysis: bool = True, 
                 enable_sampling: bool = True):
        """
        Initialize enhanced training loop
        
        Args:
            cfg: Configuration
            enable_static_analysis: Enable static analysis enhancement
            enable_sampling: Enable sampling validation
        """
        super().__init__(cfg)
        self.enable_static_analysis = enable_static_analysis
        self.enable_sampling = enable_sampling
        
        # Statistics for enhancements
        self.enhancement_stats = {
            'static_analysis_applied': 0,
            'sampling_validation_applied': 0,
            'constraints_enhanced': 0,
            'constraints_improved': 0
        }
    
    def parse_and_generate_enhanced_rules(self, raw_inferred: str, target: ErrorMessage, 
                                       arg_names: List[str], operator_name: str = None,
                                       package: str = "torch") -> List[EnhancedConstraint]:
        """
        Parse LLM response and generate enhanced rules with static analysis and sampling
        
        Args:
            raw_inferred: Raw LLM inference
            target: Error message target
            arg_names: Argument names
            operator_name: Name of the operator for static analysis
            package: Package name (torch, tensorflow)
            
        Returns:
            List of enhanced constraints
        """
        from deepconstr.train.run import parse_from_raw_txt, segment_constr
        
        generated = []
        segmented = []
        rules = []
        
        if raw_inferred is None:
            return rules
        
        infered, cot = parse_from_raw_txt(raw_inferred)
        dtypes = target.get_dtypes(arg_names)
        
        for rule_txt in infered.split(';'):
            generated.append(rule_txt.strip())
            segmented.extend(segment_constr(rule_txt.strip()))
        
        for i, rule_txts in enumerate([generated, segmented]):
            for rule_txt in rule_txts:
                if rule_txt:
                    if i == 1:  # segmented
                        cot = "divided"
                    
                    # Create enhanced constraint
                    rule = EnhancedConstraint(
                        rule_txt, cot, target, arg_names, dtypes,
                        enable_static_analysis=self.enable_static_analysis,
                        enable_sampling=self.enable_sampling
                    )
                    
                    if not rule.is_error() and rule.check():
                        # Apply enhancements
                        if operator_name:
                            enhancement_success = rule.apply_enhancements(
                                operator_name, package
                            )
                            
                            if enhancement_success:
                                self.enhancement_stats['constraints_enhanced'] += 1
                                
                                # Check if constraint was improved
                                if rule.get_enhanced_constraint_text() != rule.txt:
                                    self.enhancement_stats['constraints_improved'] += 1
                        
                        rules.append(rule)
        
        TRAIN_LOG.debug(f"Generated enhanced rules: {[c.txt for c in rules]}")
        return rules
    
    def get_enhanced_pass_rate_and_err_msgs(self, record, ntimes, operator_name: str = None,
                                          package: str = "torch") -> Tuple[float, List[ErrorMessage]]:
        """
        Get pass rate and error messages with enhanced constraint validation
        
        Args:
            record: Training record
            ntimes: Number of test iterations
            operator_name: Name of the operator
            package: Package name
            
        Returns:
            Tuple of (pass_rate, error_messages)
        """
        success_count = 0
        clusters = []
        raw_err_msgs = []
        copied_record = copy.deepcopy(record)
        
        # Convert constraints to executable format
        executable_constr = self._convert_enhanced_constr_to_executable(copied_record)
        
        # Execute with enhanced constraints
        results = self.executor.execute(
            record=copied_record,
            constraints=executable_constr,
            ntimes=ntimes,
            noise=self.cfg["train"]["noise"],
            allow_zero_length_rate=self.cfg["train"]["allow_zero_length_rate"],
            allow_zero_rate=self.cfg["train"]["allow_zero_rate"],
            num_of_try=self.cfg["train"]["num_of_try"]
        )
        
        instance_mapping = {}
        for result in results:
            if not self._is_normal_error(result):
                continue
            
            success, error_instance = result
            if success:
                success_count += 1
            else:
                msg_key = error_instance.get_core_msg()
                if instance_mapping.get(msg_key, False):
                    pass
                else:
                    instance_mapping[msg_key] = error_instance
                raw_err_msgs.append(msg_key)
        
        if raw_err_msgs:
            from deepconstr.train.errmsg import map_error_messages_to_clusters_dynamic
            dynamic_cluster_mapping = map_error_messages_to_clusters_dynamic(
                raw_err_msgs, self.cfg["str_sim_threshold"]
            )
            clusters = list(dynamic_cluster_mapping.values())
        
        pass_rate = success_count / ntimes if ntimes > 0 else 0
        return pass_rate, clusters
    
    def _convert_enhanced_constr_to_executable(self, record) -> List[Any]:
        """
        Convert enhanced constraints to executable format
        """
        from deepconstr.train.constr import convert_constr_to_executable
        
        # Get base executable constraints
        exec_rules = convert_constr_to_executable(record)
        
        # Add enhanced constraints if available
        enhanced_rules = record.get('enhanced_rules', [])
        for enhanced_rule in enhanced_rules:
            if hasattr(enhanced_rule, 'get_executable'):
                exec_rule = enhanced_rule.get_executable()
                if exec_rule is not None:
                    exec_rules.append(exec_rule)
        
        return exec_rules
    
    def _is_normal_error(self, result) -> bool:
        """
        Check if result is a normal error (not internal or timeout)
        """
        from deepconstr.train.executor import is_normal_error
        return is_normal_error(result)
    
    def train_with_enhancements(self, operator_name: str, package: str = "torch") -> Dict[str, Any]:
        """
        Train with enhanced constraints
        
        Args:
            operator_name: Name of the operator
            package: Package name
            
        Returns:
            Training results with enhancement statistics
        """
        # Get training target
        target = self.select_train_op()
        if not target:
            return {'error': 'No training target available'}
        
        # Parse and generate enhanced rules
        raw_inferred = self.synthesizer.synthesize(target)
        enhanced_rules = self.parse_and_generate_enhanced_rules(
            raw_inferred, target, target.arg_names, operator_name, package
        )
        
        if not enhanced_rules:
            return {'error': 'No enhanced rules generated'}
        
        # Evaluate enhanced rules
        evaluation_results = []
        for rule in enhanced_rules:
            # Create test record with enhanced rule
            test_record = copy.deepcopy(target.record)
            test_record['enhanced_rules'] = [rule]
            
            # Get pass rate with enhanced constraint
            pass_rate, error_messages = self.get_enhanced_pass_rate_and_err_msgs(
                test_record, self.cfg["train"]["num_eval"], operator_name, package
            )
            
            # Evaluate rule
            evaluation_result = self.evaluator.evaluate(rule, self.cfg["train"]["num_eval"])
            
            evaluation_results.append({
                'rule': rule,
                'pass_rate': pass_rate,
                'evaluation': evaluation_result,
                'enhancement_info': rule.get_enhancement_info()
            })
        
        # Update enhancement statistics
        self.enhancement_stats['static_analysis_applied'] += sum(
            1 for result in evaluation_results 
            if result['enhancement_info'].get('static_analysis_enabled', False)
        )
        self.enhancement_stats['sampling_validation_applied'] += sum(
            1 for result in evaluation_results 
            if result['enhancement_info'].get('sampling_enabled', False)
        )
        
        return {
            'enhanced_rules': enhanced_rules,
            'evaluation_results': evaluation_results,
            'enhancement_stats': self.enhancement_stats,
            'operator_name': operator_name,
            'package': package
        }
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about constraint enhancements
        """
        return {
            'enhancement_stats': self.enhancement_stats,
            'static_analysis_enabled': self.enable_static_analysis,
            'sampling_enabled': self.enable_sampling
        }
    
    def save_enhanced_constraints(self, constraints: List[EnhancedConstraint], 
                                filepath: str) -> bool:
        """
        Save enhanced constraints to file
        
        Args:
            constraints: List of enhanced constraints
            filepath: Path to save file
            
        Returns:
            True if save was successful
        """
        try:
            import json
            
            # Convert constraints to serializable format
            serializable_constraints = []
            for constraint in constraints:
                constraint_data = constraint.dump()
                serializable_constraints.append(constraint_data)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(serializable_constraints, f, indent=2)
            
            TRAIN_LOG.info(f"Saved {len(constraints)} enhanced constraints to {filepath}")
            return True
            
        except Exception as e:
            TRAIN_LOG.error(f"Failed to save enhanced constraints: {e}")
            return False
