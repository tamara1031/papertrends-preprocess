"""
Memory optimization utilities for papertrends preprocessing.
This module provides tools to monitor and manage memory usage effectively.
"""

import gc
import os
import psutil
import torch
import numpy as np
from typing import Optional, Dict, Any
import warnings

def get_memory_usage() -> Dict[str, float]:
    """Get comprehensive memory usage information."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    result = {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024,
        'total_mb': psutil.virtual_memory().total / 1024 / 1024
    }
    
    # Add GPU memory info if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024
        result.update({
            'gpu_allocated_mb': gpu_memory,
            'gpu_cached_mb': gpu_cached,
            'gpu_total_mb': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        })
    
    return result

def log_memory_usage(stage: str, verbose: bool = True) -> Dict[str, float]:
    """Log memory usage at different stages."""
    memory_info = get_memory_usage()
    
    if verbose:
        print(f"🧠 Memory usage at {stage}:")
        print(f"   • RAM: {memory_info['rss_mb']:.1f} MB ({memory_info['percent']:.1f}%)")
        print(f"   • Available: {memory_info['available_mb']:.1f} MB")
        
        if 'gpu_allocated_mb' in memory_info:
            print(f"   • GPU: {memory_info['gpu_allocated_mb']:.1f} MB allocated, "
                  f"{memory_info['gpu_cached_mb']:.1f} MB cached")
    
    return memory_info

def force_memory_cleanup(aggressive: bool = False):
    """Comprehensive memory cleanup."""
    # Python garbage collection
    collected = gc.collect()
    
    # CUDA cleanup if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if aggressive:
            # More aggressive GPU cleanup
            torch.cuda.reset_peak_memory_stats()
    
    # Additional Python garbage collection
    collected += gc.collect()
    
    if aggressive:
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
    
    return collected

def check_memory_threshold(threshold_mb: float = 1000) -> bool:
    """Check if memory usage exceeds threshold."""
    memory_info = get_memory_usage()
    return memory_info['rss_mb'] > threshold_mb

def safe_array_operation(func, *args, **kwargs):
    """Safely execute array operations with memory monitoring."""
    initial_memory = get_memory_usage()
    
    try:
        result = func(*args, **kwargs)
        
        # Check memory usage after operation
        final_memory = get_memory_usage()
        memory_increase = final_memory['rss_mb'] - initial_memory['rss_mb']
        
        if memory_increase > 500:  # More than 500MB increase
            warnings.warn(f"Large memory increase detected: {memory_increase:.1f} MB")
            force_memory_cleanup()
        
        return result
        
    except MemoryError as e:
        print(f"Memory error during operation: {e}")
        force_memory_cleanup(aggressive=True)
        raise

def optimize_numpy_memory(array: np.ndarray, target_dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Optimize numpy array memory usage."""
    if target_dtype is None:
        # Use most memory-efficient dtype that preserves data
        if array.dtype == np.float64:
            target_dtype = np.float32
        elif array.dtype == np.int64:
            target_dtype = np.int32
        else:
            return array
    
    if array.dtype != target_dtype:
        return array.astype(target_dtype)
    
    return array

def create_memory_efficient_loader(filepath: str, use_mmap: bool = True):
    """Create memory-efficient data loader."""
    if use_mmap and filepath.endswith('.npy'):
        return np.load(filepath, mmap_mode='r')
    else:
        return np.load(filepath)

def monitor_memory_during_execution(func, *args, **kwargs):
    """Monitor memory usage during function execution."""
    initial_memory = get_memory_usage()
    
    print(f"🚀 Starting execution. Initial memory: {initial_memory['rss_mb']:.1f} MB")
    
    try:
        result = func(*args, **kwargs)
        
        final_memory = get_memory_usage()
        memory_change = final_memory['rss_mb'] - initial_memory['rss_mb']
        
        print(f"✅ Execution completed. Memory change: {memory_change:+.1f} MB")
        print(f"   Final memory: {final_memory['rss_mb']:.1f} MB")
        
        return result
        
    except Exception as e:
        final_memory = get_memory_usage()
        memory_change = final_memory['rss_mb'] - initial_memory['rss_mb']
        
        print(f"❌ Execution failed. Memory change: {memory_change:+.1f} MB")
        print(f"   Final memory: {final_memory['rss_mb']:.1f} MB")
        
        raise

def get_dataset_memory_estimate(num_documents: int, embedding_dim: int = 768) -> Dict[str, float]:
    """Estimate memory requirements for dataset processing."""
    # Paper objects (rough estimate)
    paper_size_mb = num_documents * 0.001  # ~1KB per paper
    
    # Text embeddings
    embedding_size_mb = num_documents * embedding_dim * 4 / 1024 / 1024  # float32
    
    # Text strings (rough estimate)
    text_size_mb = num_documents * 0.002  # ~2KB per text
    
    # Model overhead (BERTopic + UMAP + HDBSCAN)
    model_overhead_mb = 500  # Conservative estimate
    
    total_mb = paper_size_mb + embedding_size_mb + text_size_mb + model_overhead_mb
    
    return {
        'papers_mb': paper_size_mb,
        'embeddings_mb': embedding_size_mb,
        'texts_mb': text_size_mb,
        'model_overhead_mb': model_overhead_mb,
        'total_estimated_mb': total_mb
    }

def recommend_dataset_limit(available_memory_mb: float) -> int:
    """Recommend maximum dataset size based on available memory."""
    # Conservative estimate: use only 60% of available memory
    usable_memory = available_memory_mb * 0.6
    
    # Rough calculation: ~0.5MB per document (including all overhead)
    max_documents = int(usable_memory / 0.5)
    
    return max_documents

if __name__ == "__main__":
    # Test memory utilities
    print("Memory utilities test:")
    log_memory_usage("Initial")
    
    # Create some test data
    test_array = np.random.rand(1000, 100).astype(np.float64)
    log_memory_usage("After creating test array")
    
    # Optimize array
    optimized_array = optimize_numpy_memory(test_array)
    log_memory_usage("After optimization")
    
    # Cleanup
    del test_array, optimized_array
    force_memory_cleanup()
    log_memory_usage("After cleanup")
    
    # Dataset estimation
    estimate = get_dataset_memory_estimate(20000)
    print(f"\nDataset memory estimate for 20,000 documents:")
    for key, value in estimate.items():
        print(f"  {key}: {value:.1f} MB")
