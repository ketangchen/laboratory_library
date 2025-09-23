"""
Sampling strategies for element-level constraint validation
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from abc import ABC, abstractmethod
from deepconstr.logger import CONSTR_LOG


class SamplingStrategy(ABC):
    """
    Abstract base class for sampling strategies
    """
    
    @abstractmethod
    def sample_elements(self, array: np.ndarray, sample_size: int) -> List[Any]:
        """
        Sample elements from array using the strategy
        
        Args:
            array: Array to sample from
            sample_size: Number of elements to sample
            
        Returns:
            List of sampled elements
        """
        pass


class RandomSampling(SamplingStrategy):
    """
    Random sampling strategy
    """
    
    def sample_elements(self, array: np.ndarray, sample_size: int) -> List[Any]:
        """
        Randomly sample elements from array
        """
        if array.size == 0:
            return []
        
        flat_array = array.flatten()
        
        if sample_size >= len(flat_array):
            return flat_array.tolist()
        
        indices = random.sample(range(len(flat_array)), sample_size)
        return [flat_array[i] for i in indices]


class StratifiedSampling(SamplingStrategy):
    """
    Stratified sampling strategy
    """
    
    def __init__(self, strata_count: int = 5):
        """
        Initialize stratified sampling
        
        Args:
            strata_count: Number of strata to divide elements into
        """
        self.strata_count = strata_count
    
    def sample_elements(self, array: np.ndarray, sample_size: int) -> List[Any]:
        """
        Sample elements using stratification
        """
        if array.size == 0:
            return []
        
        flat_array = array.flatten()
        
        if sample_size >= len(flat_array):
            return flat_array.tolist()
        
        # Sort array for stratification
        sorted_indices = np.argsort(flat_array)
        sorted_array = flat_array[sorted_indices]
        
        # Divide into strata
        strata_size = len(sorted_array) // self.strata_count
        sampled_elements = []
        
        for i in range(self.strata_count):
            start_idx = i * strata_size
            end_idx = start_idx + strata_size if i < self.strata_count - 1 else len(sorted_array)
            
            if start_idx < len(sorted_array):
                stratum = sorted_array[start_idx:end_idx]
                
                # Sample from this stratum
                stratum_sample_size = max(1, sample_size // self.strata_count)
                stratum_sample_size = min(stratum_sample_size, len(stratum))
                
                if stratum_sample_size < len(stratum):
                    stratum_indices = random.sample(range(len(stratum)), stratum_sample_size)
                    stratum_sample = [stratum[j] for j in stratum_indices]
                else:
                    stratum_sample = stratum.tolist()
                
                sampled_elements.extend(stratum_sample)
        
        return sampled_elements


class SystematicSampling(SamplingStrategy):
    """
    Systematic sampling strategy
    """
    
    def sample_elements(self, array: np.ndarray, sample_size: int) -> List[Any]:
        """
        Sample elements using systematic sampling
        """
        if array.size == 0:
            return []
        
        flat_array = array.flatten()
        
        if sample_size >= len(flat_array):
            return flat_array.tolist()
        
        # Calculate sampling interval
        interval = len(flat_array) // sample_size
        
        # Start at random position
        start = random.randint(0, interval - 1)
        
        # Sample elements at regular intervals
        sampled_elements = []
        for i in range(sample_size):
            idx = start + i * interval
            if idx < len(flat_array):
                sampled_elements.append(flat_array[idx])
        
        return sampled_elements


class ClusterSampling(SamplingStrategy):
    """
    Cluster sampling strategy
    """
    
    def __init__(self, cluster_size: int = 100):
        """
        Initialize cluster sampling
        
        Args:
            cluster_size: Size of each cluster
        """
        self.cluster_size = cluster_size
    
    def sample_elements(self, array: np.ndarray, sample_size: int) -> List[Any]:
        """
        Sample elements using cluster sampling
        """
        if array.size == 0:
            return []
        
        flat_array = array.flatten()
        
        if sample_size >= len(flat_array):
            return flat_array.tolist()
        
        # Calculate number of clusters needed
        num_clusters = max(1, sample_size // self.cluster_size)
        clusters_per_sample = min(num_clusters, len(flat_array) // self.cluster_size)
        
        # Randomly select clusters
        total_clusters = len(flat_array) // self.cluster_size
        if total_clusters == 0:
            # If array is smaller than cluster size, sample randomly
            return RandomSampling().sample_elements(array, sample_size)
        
        selected_clusters = random.sample(range(total_clusters), min(clusters_per_sample, total_clusters))
        
        sampled_elements = []
        for cluster_idx in selected_clusters:
            start_idx = cluster_idx * self.cluster_size
            end_idx = min(start_idx + self.cluster_size, len(flat_array))
            cluster = flat_array[start_idx:end_idx]
            sampled_elements.extend(cluster.tolist())
        
        # If we have too many elements, randomly sample from them
        if len(sampled_elements) > sample_size:
            sampled_elements = random.sample(sampled_elements, sample_size)
        
        return sampled_elements


class AdaptiveSampling(SamplingStrategy):
    """
    Adaptive sampling strategy that adjusts based on results
    """
    
    def __init__(self, initial_sample_rate: float = 0.05, max_sample_rate: float = 0.5):
        """
        Initialize adaptive sampling
        
        Args:
            initial_sample_rate: Initial sampling rate
            max_sample_rate: Maximum sampling rate
        """
        self.initial_sample_rate = initial_sample_rate
        self.max_sample_rate = max_sample_rate
    
    def sample_elements(self, array: np.ndarray, sample_size: int) -> List[Any]:
        """
        Sample elements using adaptive sampling
        """
        if array.size == 0:
            return []
        
        flat_array = array.flatten()
        
        if sample_size >= len(flat_array):
            return flat_array.tolist()
        
        # Start with initial sample rate
        current_sample_rate = self.initial_sample_rate
        current_sample_size = int(len(flat_array) * current_sample_rate)
        current_sample_size = min(current_sample_size, sample_size)
        
        # Sample elements
        if current_sample_size >= len(flat_array):
            return flat_array.tolist()
        
        indices = random.sample(range(len(flat_array)), current_sample_size)
        sampled_elements = [flat_array[i] for i in indices]
        
        return sampled_elements


class SamplingStrategyFactory:
    """
    Factory for creating sampling strategies
    """
    
    @staticmethod
    def create_strategy(strategy_name: str, **kwargs) -> SamplingStrategy:
        """
        Create sampling strategy by name
        
        Args:
            strategy_name: Name of the strategy
            **kwargs: Additional arguments for strategy
            
        Returns:
            SamplingStrategy instance
        """
        strategies = {
            'random': RandomSampling,
            'stratified': StratifiedSampling,
            'systematic': SystematicSampling,
            'cluster': ClusterSampling,
            'adaptive': AdaptiveSampling
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Unknown sampling strategy: {strategy_name}")
        
        return strategies[strategy_name](**kwargs)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """
        Get list of available sampling strategies
        
        Returns:
            List of strategy names
        """
        return ['random', 'stratified', 'systematic', 'cluster', 'adaptive']
