"""
Optimized training script demonstrating the enhanced constraint extraction
with static analysis and sampling validation
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
from omegaconf import DictConfig
from deepconstr.train.enhanced_run import EnhancedTrainingLoop
from deepconstr.static_analysis import SourceAnalyzer, ConstraintMapper, ErrorPatternMatcher
from deepconstr.sampling import ConstraintValidator, ElementSampler
from deepconstr.logger import TRAIN_LOG


class OptimizedTrainingPipeline:
    """
    Optimized training pipeline with static analysis and sampling validation
    """
    
    def __init__(self, config: DictConfig, enable_static_analysis: bool = True,
                 enable_sampling: bool = True):
        """
        Initialize optimized training pipeline
        
        Args:
            config: Training configuration
            enable_static_analysis: Enable static analysis
            enable_sampling: Enable sampling validation
        """
        self.config = config
        self.enable_static_analysis = enable_static_analysis
        self.enable_sampling = enable_sampling
        
        # Initialize components
        self.training_loop = EnhancedTrainingLoop(
            config, enable_static_analysis, enable_sampling
        )
        
        if enable_static_analysis:
            self.source_analyzer = SourceAnalyzer()
            self.constraint_mapper = ConstraintMapper()
            self.error_pattern_matcher = ErrorPatternMatcher()
        
        if enable_sampling:
            self.constraint_validator = ConstraintValidator()
        
        # Statistics
        self.pipeline_stats = {
            'operators_processed': 0,
            'constraints_generated': 0,
            'constraints_enhanced': 0,
            'static_analysis_applied': 0,
            'sampling_validation_applied': 0,
            'improvement_rate': 0.0
        }
    
    def process_operator(self, operator_name: str, package: str = "torch") -> Dict[str, Any]:
        """
        Process a single operator with optimizations
        
        Args:
            operator_name: Name of the operator
            package: Package name (torch, tensorflow)
            
        Returns:
            Processing results
        """
        TRAIN_LOG.info(f"Processing operator: {operator_name} ({package})")
        
        try:
            # Step 1: Static analysis (if enabled)
            static_analysis_result = None
            if self.enable_static_analysis:
                static_analysis_result = self._perform_static_analysis(operator_name, package)
                if static_analysis_result:
                    self.pipeline_stats['static_analysis_applied'] += 1
            
            # Step 2: Enhanced training
            training_result = self.training_loop.train_with_enhancements(operator_name, package)
            
            if 'error' in training_result:
                return training_result
            
            # Step 3: Post-processing and validation
            enhanced_rules = training_result.get('enhanced_rules', [])
            evaluation_results = training_result.get('evaluation_results', [])
            
            # Update statistics
            self.pipeline_stats['operators_processed'] += 1
            self.pipeline_stats['constraints_generated'] += len(enhanced_rules)
            self.pipeline_stats['constraints_enhanced'] += sum(
                1 for result in evaluation_results 
                if result.get('enhancement_info', {}).get('overall_confidence', 0) > 0.5
            )
            
            # Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(evaluation_results)
            
            return {
                'operator_name': operator_name,
                'package': package,
                'static_analysis_result': static_analysis_result,
                'enhanced_rules': enhanced_rules,
                'evaluation_results': evaluation_results,
                'improvement_metrics': improvement_metrics,
                'success': True
            }
            
        except Exception as e:
            TRAIN_LOG.error(f"Failed to process operator {operator_name}: {e}")
            return {
                'operator_name': operator_name,
                'package': package,
                'error': str(e),
                'success': False
            }
    
    def _perform_static_analysis(self, operator_name: str, package: str) -> Optional[Dict[str, Any]]:
        """
        Perform static analysis on operator
        """
        try:
            if package == "torch":
                return self.source_analyzer.analyze_torch_operator(operator_name)
            elif package == "tensorflow":
                return self.source_analyzer.analyze_tensorflow_operator(operator_name)
            else:
                TRAIN_LOG.warning(f"Unsupported package for static analysis: {package}")
                return None
        except Exception as e:
            TRAIN_LOG.warning(f"Static analysis failed for {operator_name}: {e}")
            return None
    
    def _calculate_improvement_metrics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate improvement metrics from evaluation results
        """
        if not evaluation_results:
            return {}
        
        # Calculate average confidence scores
        static_analysis_confidences = [
            result.get('enhancement_info', {}).get('static_analysis_confidence', 0)
            for result in evaluation_results
        ]
        sampling_confidences = [
            result.get('enhancement_info', {}).get('sampling_confidence', 0)
            for result in evaluation_results
        ]
        overall_confidences = [
            result.get('enhancement_info', {}).get('overall_confidence', 0)
            for result in evaluation_results
        ]
        
        # Calculate pass rates
        pass_rates = [
            result.get('pass_rate', 0) for result in evaluation_results
        ]
        
        return {
            'average_static_analysis_confidence': sum(static_analysis_confidences) / len(static_analysis_confidences),
            'average_sampling_confidence': sum(sampling_confidences) / len(sampling_confidences),
            'average_overall_confidence': sum(overall_confidences) / len(overall_confidences),
            'average_pass_rate': sum(pass_rates) / len(pass_rates),
            'high_confidence_rules': sum(1 for conf in overall_confidences if conf > 0.8),
            'total_rules': len(evaluation_results)
        }
    
    def batch_process_operators(self, operators: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Process multiple operators in batch
        
        Args:
            operators: List of operator dictionaries with 'name' and 'package' keys
            
        Returns:
            Batch processing results
        """
        TRAIN_LOG.info(f"Batch processing {len(operators)} operators")
        
        results = []
        successful = 0
        failed = 0
        
        for operator_info in operators:
            operator_name = operator_info.get('name')
            package = operator_info.get('package', 'torch')
            
            if not operator_name:
                TRAIN_LOG.warning("Skipping operator with missing name")
                continue
            
            result = self.process_operator(operator_name, package)
            results.append(result)
            
            if result.get('success', False):
                successful += 1
            else:
                failed += 1
        
        # Calculate overall improvement rate
        total_constraints = sum(
            len(result.get('enhanced_rules', [])) 
            for result in results if result.get('success', False)
        )
        enhanced_constraints = sum(
            result.get('improvement_metrics', {}).get('high_confidence_rules', 0)
            for result in results if result.get('success', False)
        )
        
        self.pipeline_stats['improvement_rate'] = (
            enhanced_constraints / total_constraints if total_constraints > 0 else 0
        )
        
        return {
            'results': results,
            'summary': {
                'total_operators': len(operators),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(operators) if operators else 0,
                'pipeline_stats': self.pipeline_stats
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> bool:
        """
        Save processing results to files
        
        Args:
            results: Processing results
            output_dir: Output directory
            
        Returns:
            True if save was successful
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save summary
            summary_file = os.path.join(output_dir, 'summary.json')
            with open(summary_file, 'w') as f:
                json.dump(results.get('summary', {}), f, indent=2)
            
            # Save detailed results
            detailed_file = os.path.join(output_dir, 'detailed_results.json')
            with open(detailed_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save enhanced constraints
            enhanced_constraints = []
            for result in results.get('results', []):
                if result.get('success', False):
                    for rule in result.get('enhanced_rules', []):
                        enhanced_constraints.append(rule.dump())
            
            if enhanced_constraints:
                constraints_file = os.path.join(output_dir, 'enhanced_constraints.json')
                with open(constraints_file, 'w') as f:
                    json.dump(enhanced_constraints, f, indent=2)
            
            TRAIN_LOG.info(f"Results saved to {output_dir}")
            return True
            
        except Exception as e:
            TRAIN_LOG.error(f"Failed to save results: {e}")
            return False
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics
        """
        return {
            'pipeline_stats': self.pipeline_stats,
            'static_analysis_enabled': self.enable_static_analysis,
            'sampling_enabled': self.enable_sampling
        }


def main():
    """
    Main function demonstrating optimized training pipeline
    """
    # Example usage
    config = {
        'train': {
            'num_eval': 100,
            'noise': 0.1,
            'allow_zero_length_rate': 0.1,
            'allow_zero_rate': 0.1,
            'num_of_try': 10
        },
        'str_sim_threshold': 0.5
    }
    
    # Create pipeline
    pipeline = OptimizedTrainingPipeline(
        config, 
        enable_static_analysis=True, 
        enable_sampling=True
    )
    
    # Example operators to process
    operators = [
        {'name': 'torch.add', 'package': 'torch'},
        {'name': 'torch.abs', 'package': 'torch'},
        {'name': 'torch.conv2d', 'package': 'torch'},
        {'name': 'tf.add', 'package': 'tensorflow'},
        {'name': 'tf.abs', 'package': 'tensorflow'}
    ]
    
    # Process operators
    results = pipeline.batch_process_operators(operators)
    
    # Save results
    pipeline.save_results(results, 'output/optimized_training')
    
    # Print statistics
    stats = pipeline.get_pipeline_statistics()
    print(f"Pipeline Statistics: {stats}")
    
    return results


if __name__ == "__main__":
    main()
