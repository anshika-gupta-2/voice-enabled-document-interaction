import asyncio
import time
from typing import Type, Union, Callable, Any, Optional
from functools import wraps
from loguru import logger


def retry_with_custom_backoff(
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        exceptions: tuple[Type[Exception], ...] = (Exception,),
        on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for subsequent delays
        max_delay: Maximum delay between retries in seconds
        exceptions: Tuple of exceptions to catch
        on_retry: Optional callback function called on each retry
    """

    def calculate_delay(attempt: int) -> float:
        delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
        return delay

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            last_exception = None

            while attempts < max_retries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    last_exception = e

                    if attempts == max_retries:
                        logger.error(
                            f"Final attempt {attempts}/{max_retries} "
                            f"failed for {func.__name__}: {str(e)}"
                        )
                        raise last_exception

                    delay = calculate_delay(attempts - 1)
                    if on_retry:
                        on_retry(e, attempts)

                    logger.warning(
                        f"Attempt {attempts}/{max_retries} failed "
                        f"for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)

            if last_exception:
                raise last_exception

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            last_exception = None

            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    last_exception = e

                    if attempts == max_retries:
                        logger.error(
                            f"Final attempt {attempts}/{max_retries} "
                            f"failed for {func.__name__}: {str(e)}"
                        )
                        raise last_exception

                    delay = calculate_delay(attempts - 1)
                    if on_retry:
                        on_retry(e, attempts)

                    logger.warning(
                        f"Attempt {attempts}/{max_retries} failed "
                        f"for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)

            if last_exception:
                raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
