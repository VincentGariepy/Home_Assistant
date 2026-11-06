import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dev', dest='deployment_mode', action='store_true', help='Run in development mode with dev logging')
parser.add_argument('--debug', dest='debug_mode', action='store_true', help='Run in debug mode with verbose logging')
parser.add_argument('--memory_timeframe', dest='memory_timeframe', type=int, choices=range(0, 1200), metavar="[0-1200]", help='Set memory timeframe in seconds (default: 300)', default=300)
parser.add_argument('--no_audio', dest='no_audio', action='store_true', help='Run without audio input/output (default: False)')
parser.add_argument('--no_in_audio', dest='no_in_audio', action='store_true', help='Run without audio input (default: False)')
parser.add_argument('--no_out_audio', dest='no_out_audio', action='store_true', help='Run without audio output (default: False)')
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
    no_in_audio = args.no_in_audio | args.no_audio
    no_out_audio = args.no_out_audio | args.no_audio

    assistant = HomeAssistant(memory_timeframe=memory_timeframe, no_in_audio=no_in_audio, no_out_audio=no_out_audio)
    asyncio.run(assistant.main())
