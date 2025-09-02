"""
Utility functions for the allergy detection pipeline.
"""

import pandas as pd
import numpy as np
import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Dict, Union
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)


def generate_hash(data: Union[pd.DataFrame, dict, str, list]) -> str:
    """
    Generate a hash for data to use in caching.
    
    Args:
        data: Data to hash
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, pd.DataFrame):
        # Use DataFrame shape, columns, and sample of values
        hash_data = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'sample': data.head(10).values.tobytes() if len(data) > 0 else b''
        }
        data_str = json.dumps(hash_data, sort_keys=True, default=str)
    elif isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()[:16]


def cache_result(cache_dir: Path = Path("data/cache"), ttl_hours: int = 24):
    """
    Decorator to cache function results based on input parameters.
    
    Args:
        cache_dir: Directory to store cache files
        ttl_hours: Time to live for cache in hours
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key_data = {
                'function': func.__name__,
                'args': [generate_hash(arg) if isinstance(arg, (pd.DataFrame, dict)) else str(arg) 
                        for arg in args],
                'kwargs': {k: generate_hash(v) if isinstance(v, (pd.DataFrame, dict)) else str(v) 
                          for k, v in kwargs.items()}
            }
            
            cache_key = generate_hash(cache_key_data)
            cache_file = cache_dir / f"{func.__name__}_{cache_key}.pkl"
            
            # Check if cache exists and is fresh
            if cache_file.exists():
                try:
                    cache_age = time.time() - cache_file.stat().st_mtime
                    if cache_age < ttl_hours * 3600:  # TTL in seconds
                        logger.debug(f"Loading cached result for {func.__name__}")
                        with open(cache_file, 'rb') as f:
                            return pickle.load(f)
                    else:
                        logger.debug(f"Cache expired for {func.__name__}, regenerating")
                except Exception as e:
                    logger.warning(f"Failed to load cache for {func.__name__}: {e}")
            
            # Execute function and cache result
            logger.debug(f"Computing and caching result for {func.__name__}")
            result = func(*args, **kwargs)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached result for {func.__name__}")
            except Exception as e:
                logger.warning(f"Failed to cache result for {func.__name__}: {e}")
            
            return result
        return wrapper
    return decorator


def optimize_dataframe_memory(df: pd.DataFrame, 
                             int_threshold: int = 10000,
                             float_precision: str = 'float32') -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        int_threshold: Threshold below which to use smaller int types
        float_precision: Precision for float columns ('float32' or 'float64')
        
    Returns:
        Memory-optimized DataFrame
    """
    logger.debug(f"Optimizing DataFrame memory. Original size: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type != 'object':
            # Optimize integer columns
            if 'int' in str(col_type):
                col_min = df_optimized[col].min()
                col_max = df_optimized[col].max()
                
                if col_min >= 0:  # Unsigned integers
                    if col_max < 255:
                        df_optimized[col] = df_optimized[col].astype('uint8')
                    elif col_max < 65535:
                        df_optimized[col] = df_optimized[col].astype('uint16')
                    elif col_max < int_threshold:
                        df_optimized[col] = df_optimized[col].astype('uint32')
                else:  # Signed integers
                    if col_min > -128 and col_max < 127:
                        df_optimized[col] = df_optimized[col].astype('int8')
                    elif col_min > -32768 and col_max < 32767:
                        df_optimized[col] = df_optimized[col].astype('int16')
                    elif abs(col_min) < int_threshold and col_max < int_threshold:
                        df_optimized[col] = df_optimized[col].astype('int32')
            
            # Optimize float columns
            elif 'float' in str(col_type):
                if float_precision == 'float32':
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    # Optimize object columns (convert to categories if beneficial)
    for col in df_optimized.select_dtypes(include=['object']).columns:
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # Less than 50% unique values
            df_optimized[col] = df_optimized[col].astype('category')
    
    optimized_size = df_optimized.memory_usage(deep=True).sum() / 1024**2
    original_size = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (1 - optimized_size / original_size) * 100
    
    logger.info(f"Memory optimization completed. Size: {optimized_size:.1f} MB "
               f"(reduced by {reduction:.1f}%)")
    
    return df_optimized


def parallel_apply(df: pd.DataFrame, func, axis: int = 1, n_jobs: int = -1) -> pd.Series:
    """
    Apply function to DataFrame in parallel using joblib.
    
    Args:
        df: DataFrame to process
        func: Function to apply
        axis: Axis along which to apply function
        n_jobs: Number of parallel jobs
        
    Returns:
        Series with results
    """
    try:
        from joblib import Parallel, delayed
        
        if axis == 1:  # Apply along rows
            chunks = np.array_split(df, min(len(df), abs(n_jobs) if n_jobs > 0 else 4))
            results = Parallel(n_jobs=n_jobs)(
                delayed(lambda chunk: chunk.apply(func, axis=1))(chunk) 
                for chunk in chunks
            )
            return pd.concat(results)
        else:  # Apply along columns
            results = Parallel(n_jobs=n_jobs)(
                delayed(func)(df[col]) for col in df.columns
            )
            return pd.Series(results, index=df.columns)
            
    except ImportError:
        logger.warning("joblib not available, using standard apply")
        return df.apply(func, axis=axis)


def validate_dataframe_requirements(df: pd.DataFrame, 
                                   required_columns: list,
                                   min_rows: int = 1) -> bool:
    """
    Validate that DataFrame meets requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if valid, False otherwise
    """
    if len(df) < min_rows:
        logger.error(f"DataFrame has insufficient rows: {len(df)} < {min_rows}")
        return False
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"DataFrame missing required columns: {missing_cols}")
        return False
    
    return True


def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10000):
    """
    Generator to process DataFrame in chunks.
    
    Args:
        df: DataFrame to chunk
        chunk_size: Size of each chunk
        
    Yields:
        DataFrame chunks
    """
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start:start + chunk_size]


def safe_divide(numerator: pd.Series, denominator: pd.Series, 
                fill_value: float = 0.0) -> pd.Series:
    """
    Perform safe division handling zeros and NaN values.
    
    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill_value: Value to use when division is undefined
        
    Returns:
        Series with division results
    """
    return np.where(
        (denominator == 0) | denominator.isna() | numerator.isna(),
        fill_value,
        numerator / denominator
    )


class ProgressTracker:
    """
    Simple progress tracking utility.
    """
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, increment: int = 1):
        """Update progress by increment."""
        self.current += increment
        if self.current % max(1, self.total // 20) == 0:  # Log every 5%
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            logger.info(f"{self.description}: {self.current}/{self.total} "
                       f"({100 * self.current / self.total:.1f}%) "
                       f"ETA: {eta:.0f}s")
    
    def finish(self):
        """Mark as completed."""
        elapsed = time.time() - self.start_time
        logger.info(f"{self.description} completed in {elapsed:.1f}s")