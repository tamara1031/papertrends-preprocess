"""
Memory management utilities for papertrends preprocessing.
This module provides essential memory cleanup and monitoring functions.
"""

import gc
import os
import psutil
import torch
from typing import Dict


def get_memory_usage() -> Dict[str, float]:
    """Get basic memory usage information."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    result = {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }
    
    # Add GPU memory info if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024
        result.update({
            'gpu_allocated_mb': gpu_memory,
            'gpu_cached_mb': gpu_cached
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
            torch.cuda.reset_peak_memory_stats()
    
    # Additional Python garbage collection
    collected += gc.collect()
    
    if aggressive:
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
    
    return collected


if __name__ == "__main__":
    # Test memory utilities
    print("Memory utilities test:")
    log_memory_usage("Initial")
    
    # Test cleanup
    force_memory_cleanup()
    log_memory_usage("After cleanup")