"""
Centralized logging configuration for the allergy detection pipeline.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging for the pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (created if doesn't exist)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        max_file_size: Maximum size for log files before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set up root logger
    logger = logging.getLogger('allergy_pipeline')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handlers
    if log_to_file:
        # General log file with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "pipeline.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Error-only log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        # Performance log for timing information
        perf_handler = logging.FileHandler(
            log_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(detailed_formatter)
        perf_handler.addFilter(PerfLogFilter())
        logger.addHandler(perf_handler)
    
    # Set logging level for third-party libraries to reduce noise
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('pandas').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)
    
    return logger


class PerfLogFilter(logging.Filter):
    """Filter to only include performance-related log messages."""
    
    def filter(self, record):
        # Include messages that contain timing or performance keywords
        perf_keywords = ['time', 'performance', 'duration', 'memory', 'processed', 'completed']
        return any(keyword in record.getMessage().lower() for keyword in perf_keywords)


class ContextLogger:
    """
    Context manager for logging with additional context and timing.
    """
    
    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.log(self.level, f"Completed {self.operation} in {duration.total_seconds():.2f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration.total_seconds():.2f}s: {exc_val}")
        return False  # Don't suppress exceptions


def log_memory_usage(logger: logging.Logger, operation: str = ""):
    """
    Log current memory usage if psutil is available.
    """
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage{' for ' + operation if operation else ''}: {memory_mb:.1f} MB")
    except ImportError:
        logger.debug("psutil not available for memory monitoring")
    except Exception as e:
        logger.debug(f"Error getting memory usage: {e}")


def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame"):
    """
    Log comprehensive information about a pandas DataFrame.
    """
    try:
        logger.info(f"{name} info: shape={df.shape}, memory={df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
        
        # Log data types summary
        dtypes_summary = df.dtypes.value_counts().to_dict()
        logger.debug(f"{name} data types: {dtypes_summary}")
        
        # Log missing values summary
        missing_summary = df.isnull().sum()
        if missing_summary.sum() > 0:
            missing_pct = (missing_summary / len(df) * 100).round(2)
            significant_missing = missing_pct[missing_pct > 5].to_dict()
            if significant_missing:
                logger.warning(f"{name} columns with >5% missing values: {significant_missing}")
        
    except Exception as e:
        logger.error(f"Error logging DataFrame info for {name}: {e}")