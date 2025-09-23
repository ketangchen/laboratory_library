"""
Element sampler for lightweight constraint validation
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Any, Union, Dict
from abc import ABC, abstractmethod
from deepconstr.logger import CONSTR_LOG


class ElementSampler:
    """
    Lightweight element-level constraint validation using random sampling
    """
    
    def __init__(self, sample_rate: float = 0.1, min_samples: int = 10, max_samples: int = 1000):
        """
        Initialize element sampler
        
        Args:
            sample_rate: Fraction of elements to sample (0.0 to 1.0)
            min_samples: Minimum number of samples to take
            max_samples: Maximum number of samples to take
        """
        self.sample_rate = sample_rate
        self.min_samples = min_samples
        self.max_samples = max_samples
    
    def sample_tensor_elements(self, tensor: Any, constraint_func: callable) -> Tuple[bool, Dict[str, Any]]:
        """
        Sample tensor elements and validate constraint
        
        Args:
            tensor: The tensor to sample from
            constraint_func: Function that takes element and returns bool
            
        Returns:
            Tuple of (is_valid, statistics)
        """
        try:
            # Convert tensor to numpy array for sampling
            if hasattr(tensor, 'numpy'):
                array = tensor.numpy()
            elif hasattr(tensor, 'detach'):
                array = tensor.detach().cpu().numpy()
            else:
                array = np.array(tensor)
            
            # Calculate sample size
            total_elements = array.size
            sample_size = self._calculate_sample_size(total_elements)
            
            # Sample elements
            sampled_elements = self._sample_elements(array, sample_size)
            
            # Validate constraint on sampled elements
            valid_count = 0
            invalid_elements = []
            
            for element in sampled_elements:
                try:
                    if constraint_func(element):
                        valid_count += 1
                    else:
                        invalid_elements.append(element)
                except Exception as e:
                    CONSTR_LOG.debug(f"Constraint validation failed for element {element}: {e}")
                    invalid_elements.append(element)
            
            # Calculate statistics
            is_valid = len(invalid_elements) == 0
            validity_rate = valid_count / len(sampled_elements) if sampled_elements else 0
            
            statistics = {
                'total_elements': total_elements,
                'sampled_elements': len(sampled_elements),
                'valid_elements': valid_count,
                'invalid_elements': len(invalid_elements),
                'validity_rate': validity_rate,
                'sample_rate_used': len(sampled_elements) / total_elements if total_elements > 0 else 0,
                'invalid_element_examples': invalid_elements[:5]  # Keep first 5 examples
            }
            
            return is_valid, statistics
            
        except Exception as e:
            CONSTR_LOG.warning(f"Element sampling failed: {e}")
            return False, {'error': str(e)}
    
    def _calculate_sample_size(self, total_elements: int) -> int:
        """
        Calculate appropriate sample size based on total elements and parameters
        """
        # Calculate sample size based on rate
        rate_based_size = int(total_elements * self.sample_rate)
        
        # Apply min/max constraints
        sample_size = max(self.min_samples, min(rate_based_size, self.max_samples))
        
        # Don't exceed total elements
        sample_size = min(sample_size, total_elements)
        
        return sample_size
    
    def _sample_elements(self, array: np.ndarray, sample_size: int) -> List[Any]:
        """
        Sample elements from array
        """
        if array.size == 0:
            return []
        
        # Flatten array for sampling
        flat_array = array.flatten()
        
        if sample_size >= len(flat_array):
            return flat_array.tolist()
        
        # Random sampling
        indices = random.sample(range(len(flat_array)), sample_size)
        sampled_elements = [flat_array[i] for i in indices]
        
        return sampled_elements
    
    def sample_with_stratification(self, tensor: Any, constraint_func: callable, 
                                 strata_count: int = 5) -> Tuple[bool, Dict[str, Any]]:
        """
        Sample elements with stratification for better coverage
        
        Args:
            tensor: The tensor to sample from
            constraint_func: Function that takes element and returns bool
            strata_count: Number of strata to divide elements into
            
        Returns:
            Tuple of (is_valid, statistics)
        """
        try:
            # Convert tensor to numpy array
            if hasattr(tensor, 'numpy'):
                array = tensor.numpy()
            elif hasattr(tensor, 'detach'):
                array = tensor.detach().cpu().numpy()
            else:
                array = np.array(tensor)
            
            if array.size == 0:
                return True, {'total_elements': 0, 'sampled_elements': 0}
            
            # Flatten and sort for stratification
            flat_array = array.flatten()
            sorted_indices = np.argsort(flat_array)
            sorted_array = flat_array[sorted_indices]
            
            # Divide into strata
            strata_size = len(sorted_array) // strata_count
            strata = []
            
            for i in range(strata_count):
                start_idx = i * strata_size
                end_idx = start_idx + strata_size if i < strata_count - 1 else len(sorted_array)
                strata.append(sorted_array[start_idx:end_idx])
            
            # Sample from each stratum
            all_sampled = []
            stratum_stats = []
            
            for i, stratum in enumerate(strata):
                if len(stratum) == 0:
                    continue
                
                # Sample from this stratum
                stratum_sample_size = max(1, len(stratum) // 10)  # 10% of stratum
                stratum_sample_size = min(stratum_sample_size, len(stratum))
                
                if stratum_sample_size < len(stratum):
                    stratum_indices = random.sample(range(len(stratum)), stratum_sample_size)
                    stratum_sample = [stratum[j] for j in stratum_indices]
                else:
                    stratum_sample = stratum.tolist()
                
                all_sampled.extend(stratum_sample)
                stratum_stats.append({
                    'stratum': i,
                    'size': len(stratum),
                    'sampled': len(stratum_sample)
                })
            
            # Validate constraint on all sampled elements
            valid_count = 0
            invalid_elements = []
            
            for element in all_sampled:
                try:
                    if constraint_func(element):
                        valid_count += 1
                    else:
                        invalid_elements.append(element)
                except Exception as e:
                    CONSTR_LOG.debug(f"Constraint validation failed for element {element}: {e}")
                    invalid_elements.append(element)
            
            # Calculate statistics
            is_valid = len(invalid_elements) == 0
            validity_rate = valid_count / len(all_sampled) if all_sampled else 0
            
            statistics = {
                'total_elements': array.size,
                'sampled_elements': len(all_sampled),
                'valid_elements': valid_count,
                'invalid_elements': len(invalid_elements),
                'validity_rate': validity_rate,
                'strata_count': strata_count,
                'stratum_stats': stratum_stats,
                'invalid_element_examples': invalid_elements[:5]
            }
            
            return is_valid, statistics
            
        except Exception as e:
            CONSTR_LOG.warning(f"Stratified sampling failed: {e}")
            return False, {'error': str(e)}
    
    def adaptive_sampling(self, tensor: Any, constraint_func: callable, 
                         initial_sample_rate: float = 0.05,
                         max_iterations: int = 3) -> Tuple[bool, Dict[str, Any]]:
        """
        Adaptive sampling that adjusts sample rate based on initial results
        
        Args:
            tensor: The tensor to sample from
            constraint_func: Function that takes element and returns bool
            initial_sample_rate: Initial sampling rate
            max_iterations: Maximum number of sampling iterations
            
        Returns:
            Tuple of (is_valid, statistics)
        """
        try:
            # Convert tensor to numpy array
            if hasattr(tensor, 'numpy'):
                array = tensor.numpy()
            elif hasattr(tensor, 'detach'):
                array = tensor.detach().cpu().numpy()
            else:
                array = np.array(tensor)
            
            if array.size == 0:
                return True, {'total_elements': 0, 'sampled_elements': 0}
            
            # Start with initial sample rate
            current_sample_rate = initial_sample_rate
            all_sampled = []
            iteration_stats = []
            
            for iteration in range(max_iterations):
                # Calculate sample size for this iteration
                total_elements = array.size
                sample_size = int(total_elements * current_sample_rate)
                sample_size = max(self.min_samples, min(sample_size, self.max_samples))
                sample_size = min(sample_size, total_elements)
                
                # Sample elements
                flat_array = array.flatten()
                if sample_size >= len(flat_array):
                    iteration_sample = flat_array.tolist()
                else:
                    indices = random.sample(range(len(flat_array)), sample_size)
                    iteration_sample = [flat_array[i] for i in indices]
                
                all_sampled.extend(iteration_sample)
                
                # Validate constraint
                valid_count = 0
                invalid_elements = []
                
                for element in iteration_sample:
                    try:
                        if constraint_func(element):
                            valid_count += 1
                        else:
                            invalid_elements.append(element)
                    except Exception as e:
                        CONSTR_LOG.debug(f"Constraint validation failed for element {element}: {e}")
                        invalid_elements.append(element)
                
                iteration_validity_rate = valid_count / len(iteration_sample) if iteration_sample else 0
                
                iteration_stats.append({
                    'iteration': iteration + 1,
                    'sample_rate': current_sample_rate,
                    'sample_size': len(iteration_sample),
                    'validity_rate': iteration_validity_rate,
                    'invalid_count': len(invalid_elements)
                })
                
                # If we found invalid elements, increase sample rate for next iteration
                if invalid_elements:
                    current_sample_rate = min(current_sample_rate * 2, 0.5)  # Cap at 50%
                else:
                    # If all valid, we can stop or reduce sample rate
                    break
            
            # Final validation on all sampled elements
            final_valid_count = 0
            final_invalid_elements = []
            
            for element in all_sampled:
                try:
                    if constraint_func(element):
                        final_valid_count += 1
                    else:
                        final_invalid_elements.append(element)
                except Exception as e:
                    CONSTR_LOG.debug(f"Constraint validation failed for element {element}: {e}")
                    final_invalid_elements.append(element)
            
            # Calculate final statistics
            is_valid = len(final_invalid_elements) == 0
            validity_rate = final_valid_count / len(all_sampled) if all_sampled else 0
            
            statistics = {
                'total_elements': array.size,
                'sampled_elements': len(all_sampled),
                'valid_elements': final_valid_count,
                'invalid_elements': len(final_invalid_elements),
                'validity_rate': validity_rate,
                'iterations': len(iteration_stats),
                'iteration_stats': iteration_stats,
                'invalid_element_examples': final_invalid_elements[:5]
            }
            
            return is_valid, statistics
            
        except Exception as e:
            CONSTR_LOG.warning(f"Adaptive sampling failed: {e}")
            return False, {'error': str(e)}
