import sys
import os
import platform
import asyncio
from functools import wraps

from audio_functions.genai_audio_utils import convert_text_to_speech

def async_resilient(retries=3, timeout=5):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    # Call the async function with timeout
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                    return result
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
        return async_wrapper
    return decorator

def extract_last_ai_message(raw_string):
    # Mapping the class names to generic dictionaries so Python can read them
    class_map = {
        'HumanMessage': lambda **kwargs: kwargs,
        'AIMessage': lambda **kwargs: kwargs,
        'ToolMessage': lambda **kwargs: kwargs,
    }
    
    # Parse the string into a Python list
    try:
        data = eval(raw_string, {"__builtins__": None}, class_map)
        
        # Get the last item
        last_msg = data[-1]
        
        # The content is a list: [{'type': 'text', 'text': '...'}]
        # We grab the 'text' field from the first element
        if isinstance(last_msg['content'], list):
            return last_msg['content'][0].get('text')
        return last_msg['content']
        
    except Exception as e:
        return f"Parsing error: {e}"

def check_environment():
    # 1. Check if running on Linux
    is_linux = sys.platform.startswith('linux')
    
    # 2. Check if running specifically on a Raspberry Pi
    is_raspberry_pi = False
    if is_linux:
        # Method A: Read the hardware model from the Linux device tree
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi' in model:
                    is_raspberry_pi = True
        
        # Method B: Fallback check via uname/platform
        elif 'raspberrypi' in platform.uname().node.lower():
            is_raspberry_pi = True

    return is_linux, is_raspberry_pi