import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dev', dest='deployment_mode', action='store_true', help='Run in development mode with dev logging')
parser.add_argument('--debug', dest='debug_mode', action='store_true', help='Run in debug mode with verbose logging')
parser.add_argument('--memory_timeframe', dest='memory_timeframe', type=int, choices=range(0, 121), metavar="[0-120]", help='Set memory timeframe in minutes (default: 5)', default=5)
parser.add_argument('--no_audio', dest='no_audio', action='store_true', help='Run without audio input/output (default: False)')
parser.add_argument('--quirky_response_interval', dest='quirky_response_interval', type=int, choices=[-1, 3, 5, 10, 15], help='Set interval for quirky responses in minutes, -1 means never (default: 10)', default=10)

args = parser.parse_args()

import asyncio
from log_config.logging_config import get_conversation_logger, set_debug_mode
_ = get_conversation_logger(dev=args.deployment_mode)

# Set debug mode before importing HomeAssistant so debug logs are captured
if args.debug_mode:
    set_debug_mode(debug=True)

from agents.home_assistant import HomeAssistant

if __name__ == "__main__":
    memory_timeframe = args.memory_timeframe
    no_audio = args.no_audio
    quirky_response_interval = None if args.quirky_response_interval == -1 else args.quirky_response_interval
   
    assistant = HomeAssistant(memory_timeframe=memory_timeframe, no_audio=no_audio, quirky_response_interval=quirky_response_interval)
    asyncio.run(assistant.main_2())