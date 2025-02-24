import logging
import time
from functools import wraps


def time_counter(logger_name='default_logger'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取指定的 logger
            logger = logging.getLogger(logger_name)

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            elapsed_time_seconds = end_time - start_time
            elapsed_time_minutes = int(elapsed_time_seconds // 60)
            elapsed_time_seconds = int(elapsed_time_seconds % 60)

            logger.info(f"Function [{func.__name__}] cost: {elapsed_time_minutes} min {elapsed_time_seconds} sec.")

            return result

        return wrapper

    return decorator
