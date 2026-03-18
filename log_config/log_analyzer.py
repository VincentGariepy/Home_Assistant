# This file is 100% Vibe Coded
"""
Log analysis utility for Home Assistant conversations and token usage.

Usage:
    python log_analyzer.py # Create file with all stats
    python log_analyzer.py --conversations     # Create file with recent conversations
    python log_analyzer.py --tokens            # Create file with token usage statistics
    python log_analyzer.py --errors            # Create file with recent errors
    python log_analyzer.py --stats             # Create file with overall statistics
    python log_analyzer.py --conversation-id ID  # Create file with specific conversation

    *Add the --dev flag to analyze dev logs instead of production logs.
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Optional, TextIO

LOGS_DIR = Path("logs")
ANALYSIS_DIR = Path("logging/analysis")

class OutputManager:
    """Manages output to both stdout and file."""
    
    def __init__(self, use_file: bool = False):
        self.use_file = use_file
        self.file_handle: Optional[TextIO] = None
        self.file_path: Optional[Path] = None
        
        if use_file:
            ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.file_path = ANALYSIS_DIR / f"analysis_{timestamp}.txt"
            self.file_handle = open(self.file_path, 'w')
    
    def write(self, text: str = ""):
        """Write to stdout and optionally to file."""
        print(text)
        if self.file_handle:
            self.file_handle.write(text + "\n")
    
    def close(self):
        """Close the file if open."""
        if self.file_handle:
            self.file_handle.close()
            print(f"\nAnalysis saved to: {self.file_path}", file=sys.stderr)

# Global output manager
output_manager: Optional[OutputManager] = None

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of entries."""
    if not file_path.exists():
        return []
    
    entries = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries

def format_timestamp(iso_string: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_string

def filter_data(data: List[Dict[str, Any]], date_filter: str = None) -> List[Dict[str, Any]]:
    """Filter data by date. Shows entries from this date onward."""
    if not date_filter:
        return data
    
    try:
        # Handle both YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS formats
        if len(date_filter) == 10:  # YYYY-MM-DD
            filter_dt = datetime.fromisoformat(date_filter + "T00:00:00")
        else:
            filter_dt = datetime.fromisoformat(date_filter)
        
        filtered = [item for item in data if datetime.fromisoformat(item.get('timestamp', '')) >= filter_dt]
        return filtered
    except ValueError:
        print(f"Invalid date format: {date_filter}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
        return data

def show_conversations(limit: int = 10, date_filter: str = None, dev: bool = False):
    """Display recent conversations."""
    log_file = "dev_conversations.jsonl" if dev else "conversations.jsonl"
    conversations = load_jsonl(LOGS_DIR / log_file)
    
    if not conversations:
        output_manager.write("No conversations logged yet.")
        return
    
    conversations = filter_data(conversations, date_filter)
    
    if not conversations:
        output_manager.write("No conversations found for the specified date.")
        return
    
    output_manager.write(f"\n{'='*80}")
    output_manager.write(f"RECENT CONVERSATIONS (showing last {min(limit, len(conversations))}) {'[DEV]' if dev else '[PROD]'}")
    if date_filter:
        output_manager.write(f"From: {date_filter}")
    output_manager.write(f"{'='*80}\n")
    
    for conv in conversations[-limit:]:
        output_manager.write(f"Time: {format_timestamp(conv.get('timestamp', 'N/A'))}")
        output_manager.write(f"Wake Word: {conv.get('wake_word', 'N/A')}")
        output_manager.write(f"Tone: {conv.get('tone', 'N/A')}")
        output_manager.write(f"User: {conv.get('user_input', 'N/A')[:80]}...")
        output_manager.write(f"Assistant: {conv.get('assistant_response', 'N/A')[:80]}...")
        output_manager.write("-" * 80)

def show_token_usage_stats(date_filter: str = None, dev: bool = False):
    """Display token usage statistics."""
    log_file = "dev_token_usage.jsonl" if dev else "token_usage.jsonl"
    token_logs = load_jsonl(LOGS_DIR / log_file)
    
    if not token_logs:
        output_manager.write("No token usage data logged yet.")
        return
    
    token_logs = filter_data(token_logs, date_filter)
    
    if not token_logs:
        output_manager.write("No token usage data found for the specified date.")
        return
    
    output_manager.write(f"\n{'='*80}")
    output_manager.write(f"TOKEN USAGE STATISTICS {'[DEV]' if dev else '[PROD]'}")
    if date_filter:
        output_manager.write(f"From: {date_filter}")
    output_manager.write(f"{'='*80}\n")
    
    # Group by agent
    stats_by_agent = defaultdict(lambda: {
        'calls': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
    })
    
    for log in token_logs:
        agent = log.get('agent_name', 'unknown')
        stats_by_agent[agent]['calls'] += 1
        stats_by_agent[agent]['input_tokens'] += log.get('input_tokens', 0)
        stats_by_agent[agent]['output_tokens'] += log.get('output_tokens', 0)
        stats_by_agent[agent]['total_tokens'] += log.get('total_tokens', 0)
    
    # Display per-agent stats
    output_manager.write(f"{'Agent':<25} {'Calls':<10} {'Input':<12} {'Output':<12} {'Total':<12}")
    output_manager.write("-" * 71)
    
    total_calls = 0
    total_input = 0
    total_output = 0
    total_tokens = 0
    
    for agent in sorted(stats_by_agent.keys()):
        stats = stats_by_agent[agent]
        output_manager.write(f"{agent:<25} {stats['calls']:<10} {stats['input_tokens']:<12} "
              f"{stats['output_tokens']:<12} {stats['total_tokens']:<12}")
        
        total_calls += stats['calls']
        total_input += stats['input_tokens']
        total_output += stats['output_tokens']
        total_tokens += stats['total_tokens']
    
    output_manager.write("-" * 71)
    output_manager.write(f"{'TOTAL':<25} {total_calls:<10} {total_input:<12} {total_output:<12} {total_tokens:<12}")
    output_manager.write()

def show_errors(limit: int = 10, date_filter: str = None, dev: bool = False):
    """Display recent errors."""
    log_file = "dev_errors.jsonl" if dev else "errors.jsonl"
    errors = load_jsonl(LOGS_DIR / log_file)
    
    if not errors:
        output_manager.write("No errors logged yet.")
        return
    
    errors = filter_data(errors, date_filter)
    
    if not errors:
        output_manager.write("No errors found for the specified date.")
        return
    
    output_manager.write(f"\n{'='*80}")
    output_manager.write(f"RECENT ERRORS (showing last {min(limit, len(errors))}) {'[DEV]' if dev else '[PROD]'}")
    if date_filter:
        output_manager.write(f"From: {date_filter}")
    output_manager.write(f"{'='*80}\n")
    
    for error in errors[-limit:]:
        output_manager.write(f"Time: {format_timestamp(error.get('timestamp', 'N/A'))}")
        output_manager.write(f"Error Type: {error.get('error_type', 'N/A')}")
        output_manager.write(f"Message: {error.get('error_message', 'N/A')}")
        if error.get('context'):
            output_manager.write(f"Context: {error.get('context')}")
        output_manager.write("-" * 80)

def show_overall_stats(date_filter: str = None, dev: bool = False):
    """Display overall statistics."""
    log_prefix = "dev_" if dev else ""
    conversations = load_jsonl(LOGS_DIR / f"{log_prefix}conversations.jsonl")
    token_logs = load_jsonl(LOGS_DIR / f"{log_prefix}token_usage.jsonl")
    errors = load_jsonl(LOGS_DIR / f"{log_prefix}errors.jsonl")
    
    conversations = filter_data(conversations, date_filter)
    token_logs = filter_data(token_logs, date_filter)
    errors = filter_data(errors, date_filter)
    
    output_manager.write(f"\n{'='*80}")
    output_manager.write(f"OVERALL STATISTICS {'[DEV]' if dev else '[PROD]'}")
    if date_filter:
        output_manager.write(f"From: {date_filter}")
    output_manager.write(f"{'='*80}\n")
    
    # Calculate time range
    if conversations:
        first_time = datetime.fromisoformat(conversations[0]['timestamp'])
        last_time = datetime.fromisoformat(conversations[-1]['timestamp'])
        duration = last_time - first_time
        output_manager.write(f"Total Conversations: {len(conversations)}")
        output_manager.write(f"Time Span: {format_timestamp(first_time)} to {format_timestamp(last_time)}")
        output_manager.write(f"Duration: {duration}")
    else:
        output_manager.write("Total Conversations: 0")
    
    # Token statistics
    if token_logs:
        total_tokens = sum(log.get('total_tokens', 0) for log in token_logs)
        output_manager.write(f"\nTotal LLM Calls: {len(token_logs)}")
        output_manager.write(f"Total Tokens Used: {total_tokens:,}")
        avg_tokens = total_tokens / len(token_logs) if token_logs else 0
        output_manager.write(f"Average Tokens per Call: {avg_tokens:.0f}")
    else:
        output_manager.write("\nTotal LLM Calls: 0")
    
    # Tone distribution
    if conversations:
        tone_counts = defaultdict(int)
        for conv in conversations:
            tone = conv.get('tone', 'unknown')
            tone_counts[tone] += 1
        output_manager.write(f"\nTone Distribution:")
        for tone, count in sorted(tone_counts.items()):
            output_manager.write(f"  {tone}: {count}")
    
    # Error statistics
    if errors:
        error_types = defaultdict(int)
        for error in errors:
            error_type = error.get('error_type', 'unknown')
            error_types[error_type] += 1
        output_manager.write(f"\nTotal Errors: {len(errors)}")
        output_manager.write("Error Types:")
        for error_type, count in sorted(error_types.items()):
            output_manager.write(f"  {error_type}: {count}")
    else:
        output_manager.write("\nTotal Errors: 0")
    
    output_manager.write()

def show_conversation_by_id(conversation_id: str, dev: bool = False):
    """Show all messages from a specific conversation."""
    log_file = "dev_conversations.jsonl" if dev else "conversations.jsonl"
    conversations = load_jsonl(LOGS_DIR / log_file)
    
    matching = [c for c in conversations if c.get('conversation_id') == conversation_id]
    
    if not matching:
        output_manager.write(f"No conversations found with ID: {conversation_id}")
        return
    
    output_manager.write(f"\n{'='*80}")
    output_manager.write(f"CONVERSATION: {conversation_id} {'[DEV]' if dev else '[PROD]'}")
    output_manager.write(f"{'='*80}\n")
    
    for i, conv in enumerate(matching, 1):
        output_manager.write(f"Message #{i}")
        output_manager.write(f"Time: {format_timestamp(conv.get('timestamp', 'N/A'))}")
        output_manager.write(f"Wake Word: {conv.get('wake_word', 'N/A')}")
        output_manager.write(f"Tone: {conv.get('tone', 'N/A')}")
        output_manager.write(f"\nUser:\n{conv.get('user_input', 'N/A')}")
        output_manager.write(f"\nAssistant:\n{conv.get('assistant_response', 'N/A')}")
        output_manager.write("\n" + "-" * 80)

def main():
    global output_manager
    
    parser = argparse.ArgumentParser(description="Analyze Home Assistant logs")
    parser.add_argument("--conversations", action="store_true", help="Show recent conversations")
    parser.add_argument("--tokens", action="store_true", help="Show token usage statistics")
    parser.add_argument("--errors", action="store_true", help="Show recent errors")
    parser.add_argument("--stats", action="store_true", help="Show overall statistics")
    parser.add_argument("--conversation-id", type=str, help="Show specific conversation by ID")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of items to show (default: 10)")
    parser.add_argument("--from-date", type=str, help="Analyze data from this date onward (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    parser.add_argument("--save", action="store_true", help="Save analysis to a dated file")
    parser.add_argument("--dev", action="store_true", help="Analyze dev logs instead of production logs")
    
    args = parser.parse_args()
    
    # Initialize output manager
    output_manager = OutputManager(use_file=args.save)
    
    # If no arguments provided, show all
    if not any([args.conversations, args.tokens, args.errors, args.stats, args.conversation_id]):
        args.stats = True
        args.conversations = True
        args.tokens = True
    
    try:
        if args.conversations:
            show_conversations(args.limit, args.from_date, args.dev)
        
        if args.tokens:
            show_token_usage_stats(args.from_date, args.dev)
        
        if args.errors:
            show_errors(args.limit, args.from_date, args.dev)
        
        if args.stats:
            show_overall_stats(args.from_date, args.dev)
        
        if args.conversation_id:
            show_conversation_by_id(args.conversation_id, args.dev)
    finally:
        output_manager.close()

if __name__ == "__main__":
    main()
