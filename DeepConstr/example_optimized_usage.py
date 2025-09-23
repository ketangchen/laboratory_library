#!/usr/bin/env python3
"""
Example script demonstrating the optimized DeepConstr training
with static analysis and sampling validation
"""

import sys
import os
import json
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepconstr.integration.optimized_training import OptimizedTrainingPipeline
from deepconstr.static_analysis import SourceAnalyzer, ConstraintMapper, ErrorPatternMatcher
from deepconstr.sampling import ConstraintValidator, ElementSampler
from deepconstr.logger import TRAIN_LOG


def demonstrate_static_analysis():
    """
    Demonstrate static analysis capabilities
    """
    print("=== Static Analysis Demonstration ===")
    
    # Initialize static analysis components
    source_analyzer = SourceAnalyzer()
    constraint_mapper = ConstraintMapper()
    error_pattern_matcher = ErrorPatternMatcher()
    
    # Example error messages
    error_messages = [
        "kernel size too large",
        "negative dimensions are not allowed",
        "shape mismatch",
        "element exceeds maximum value"
    ]
    
    print("Analyzing error messages...")
    for error_msg in error_messages:
        # Pattern matching
        pattern_result = error_pattern_matcher.analyze_error_message(error_msg)
        print(f"Error: {error_msg}")
        print(f"  Pattern matched: {pattern_result['matched']}")
        if pattern_result['matched']:
            print(f"  Constraint: {pattern_result['constraint']}")
            print(f"  Confidence: {pattern_result['confidence']:.2f}")
        print()
    
    # Example constraint mapping
    print("Mapping error to constraint...")
    error_msg = "kernel size too large"
    operator_name = "torch.conv2d"
    
    # Simulate static analysis result
    static_analysis_result = {
        'constraints': [
            {'type': 'comparison', 'left': 'kernel_size', 'operator': '<=', 'right': 'input_size + 2 * padding', 'line': 10}
        ],
        'error_mappings': {
            'kernel size too large': ['kernel_size <= input_size + 2 * padding']
        }
    }
    
    enhanced_mapping = constraint_mapper.create_enhanced_error_mapping(
        error_msg, operator_name, static_analysis_result
    )
    
    print(f"Error: {error_msg}")
    print(f"Enhanced mapping: {enhanced_mapping}")
    print()


def demonstrate_sampling_validation():
    """
    Demonstrate sampling validation capabilities
    """
    print("=== Sampling Validation Demonstration ===")
    
    # Initialize sampling components
    sampler = ElementSampler(sample_rate=0.1, min_samples=10, max_samples=100)
    validator = ConstraintValidator(sampler)
    
    # Example tensor data
    import numpy as np
    test_tensor = np.random.rand(1000, 1000) * 10  # Random tensor with values 0-10
    
    print(f"Testing tensor with shape: {test_tensor.shape}")
    print(f"Tensor value range: {test_tensor.min():.2f} to {test_tensor.max():.2f}")
    
    # Test different constraints
    constraints = [
        "element <= 5",
        "element <= 8",
        "element <= 12"
    ]
    
    for constraint in constraints:
        print(f"\nTesting constraint: {constraint}")
        
        # Random sampling
        is_valid, stats = validator.validate_element_constraint(
            test_tensor, constraint
        )
        
        print(f"  Random sampling - Valid: {is_valid}")
        print(f"  Sample rate: {stats.get('sample_rate_used', 0):.3f}")
        print(f"  Validity rate: {stats.get('validity_rate', 0):.3f}")
        print(f"  Sampled elements: {stats.get('sampled_elements', 0)}")
        
        # Stratified sampling
        is_valid_stratified, stats_stratified = validator.validate_element_constraint_stratified(
            test_tensor, constraint, strata_count=5
        )
        
        print(f"  Stratified sampling - Valid: {is_valid_stratified}")
        print(f"  Validity rate: {stats_stratified.get('validity_rate', 0):.3f}")
        print(f"  Strata count: {stats_stratified.get('strata_count', 0)}")
    
    print()


def demonstrate_enhanced_training():
    """
    Demonstrate enhanced training pipeline
    """
    print("=== Enhanced Training Pipeline Demonstration ===")
    
    # Configuration
    config = {
        'train': {
            'num_eval': 50,
            'noise': 0.1,
            'allow_zero_length_rate': 0.1,
            'allow_zero_rate': 0.1,
            'num_of_try': 5
        },
        'str_sim_threshold': 0.7
    }
    
    # Create pipeline
    pipeline = OptimizedTrainingPipeline(
        config,
        enable_static_analysis=True,
        enable_sampling=True
    )
    
    # Example operators
    operators = [
        {'name': 'torch.add', 'package': 'torch'},
        {'name': 'torch.abs', 'package': 'torch'},
        {'name': 'torch.conv2d', 'package': 'torch'}
    ]
    
    print("Processing operators with enhanced pipeline...")
    
    # Process each operator
    results = []
    for operator in operators:
        print(f"\nProcessing {operator['name']} ({operator['package']})...")
        
        result = pipeline.process_operator(operator['name'], operator['package'])
        results.append(result)
        
        if result.get('success', False):
            print(f"  Success: Generated {len(result.get('enhanced_rules', []))} enhanced rules")
            
            # Show enhancement info for first rule
            enhanced_rules = result.get('enhanced_rules', [])
            if enhanced_rules:
                first_rule = enhanced_rules[0]
                enhancement_info = first_rule.get_enhancement_info()
                print(f"  Enhancement info:")
                print(f"    Original: {enhancement_info.get('original_constraint', 'N/A')}")
                print(f"    Enhanced: {enhancement_info.get('enhanced_constraint', 'N/A')}")
                print(f"    Confidence: {enhancement_info.get('overall_confidence', 0):.3f}")
        else:
            print(f"  Failed: {result.get('error', 'Unknown error')}")
    
    # Show pipeline statistics
    stats = pipeline.get_pipeline_statistics()
    print(f"\nPipeline Statistics:")
    print(f"  Operators processed: {stats['pipeline_stats']['operators_processed']}")
    print(f"  Constraints generated: {stats['pipeline_stats']['constraints_generated']}")
    print(f"  Constraints enhanced: {stats['pipeline_stats']['constraints_enhanced']}")
    print(f"  Improvement rate: {stats['pipeline_stats']['improvement_rate']:.3f}")
    
    return results


def demonstrate_optimization_benefits():
    """
    Demonstrate the benefits of the optimizations
    """
    print("=== Optimization Benefits Demonstration ===")
    
    print("1. Static Analysis Benefits:")
    print("   - Direct analysis of operator source code")
    print("   - Precise error message to constraint mapping")
    print("   - Avoids ambiguity in error message interpretation")
    print("   - Example: 'kernel_size too large' -> 'kernel_size <= input_size + 2 * padding'")
    print()
    
    print("2. Sampling Validation Benefits:")
    print("   - Lightweight element-level constraint validation")
    print("   - 10% sampling rate vs 100% element checking")
    print("   - Significant performance improvement for large tensors")
    print("   - Maintains high accuracy with statistical confidence")
    print()
    
    print("3. Combined Benefits:")
    print("   - More accurate constraint extraction")
    print("   - Faster validation process")
    print("   - Better error message understanding")
    print("   - Improved overall system performance")
    print()


def main():
    """
    Main demonstration function
    """
    print("DeepConstr Optimization Demonstration")
    print("=" * 50)
    
    try:
        # Demonstrate static analysis
        demonstrate_static_analysis()
        
        # Demonstrate sampling validation
        demonstrate_sampling_validation()
        
        # Demonstrate enhanced training
        results = demonstrate_enhanced_training()
        
        # Show optimization benefits
        demonstrate_optimization_benefits()
        
        print("Demonstration completed successfully!")
        
        # Save results if available
        if results:
            output_dir = "output/demonstration"
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, "results.json"), 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
