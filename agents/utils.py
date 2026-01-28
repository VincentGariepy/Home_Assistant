import asyncio
import inspect
from functools import wraps

from agents.audio_agent import convert_text_to_speech

def async_resilient(retries=3, timeout=5):
    def decorator(func):
        @wraps(func)
        async def async_gen_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    # Create async generator
                    async_gen = func(*args, **kwargs)
                    start_time = asyncio.get_event_loop().time()
                    async for item in async_gen:
                        # Check if we've exceeded the timeout
                        if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                            raise asyncio.TimeoutError("Async generator exceeded timeout")
                        yield item
                    return
                except asyncio.TimeoutError:
                    print(f"Attempt {attempt + 1} timed out.")
                    if attempt == retries - 1:
                        await convert_text_to_speech("Oops, looks like there is an timeout, no more retrying", tone='neutral')
                    else:
                        await convert_text_to_speech("Oops, looks like there is a timeout, retrying.", tone='neutral')
                    last_exception = TimeoutError("Function execution exceeded limit")
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed with: {e}")
                    last_exception = e
                    if attempt == retries - 1:
                        await convert_text_to_speech("Oops, looks like there is an error, no more retrying", tone='neutral')
                    else:
                        await convert_text_to_speech("Oops, looks like there is an error, retrying.", tone='neutral')
                
                # Exponential backoff: 1s, 2s, 4s...
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
            
            raise last_exception
        return async_gen_wrapper
    return decorator