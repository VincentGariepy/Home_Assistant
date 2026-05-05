# This file is 100% Vibe Coded
"""
Log analysis utility for Home Assistant conversations and token usage.

Usage:
    python log_analyzer.py # Output analysis with all stats
    python log_analyzer.py --conversations     # Output analysis with recent conversations
    python log_analyzer.py --tokens            # Output analysis with token usage statistics
    python log_analyzer.py --errors            # Output analysis with recent errors
    python log_analyzer.py --stats             # Output analysis with overall statistics
    python log_analyzer.py --conversation-id ID  # Output analysis with specific conversation

    * Add the --dev flag to analyze dev logs instead of production logs.
    * Add the --from-date YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS flag to filter logs from a specific date onward.
    * Add the --save flag to save the analysis to a dated file in logging/analysis
    * Add the --email flag to email the analysis results instead of printing
"""

import json
import argparse
import os
import sys
import smtplib
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Optional, TextIO
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

LOGS_DIR = Path("logs")
ANALYSIS_DIR = Path("log_config/analysis")
MONTHLY_EMAIL_PATH = Path("log_config/monthly_email.txt")

EMAIL_FROM = 'vgariepy339@gmail.com'
EMAIL_TO = 'vincent.gariepy@gmail.com'
EMAIL_PASSWORD = os.getenv('GMAIL_VGARIEPY339_APP_PASSWORD')

with open('agents/agents_config.yaml', 'r') as file:
    AGENT_CONFIG = yaml.safe_load(file)

with open('log_config/model_costs.yaml', 'r') as file:
    COST_CONFIG = yaml.safe_load(file)

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

def filter_data(data: List[Dict[str, Any]], date_filter: tuple[str, str] = None) -> List[Dict[str, Any]]:
    """Filter data by date. Shows entries from this date onward."""
    if not date_filter:
        return data
    
    try:
        if not date_filter[0]:
            initial_date = datetime.min
        else:
            initial_date = datetime.fromisoformat(date_filter[0] + "T00:00:00")
        
        if not date_filter[1]:
            final_date = datetime.max
        else:
            final_date = datetime.fromisoformat(date_filter[1] + "T00:00:00")
        
        filtered = [item for item in data if initial_date <= datetime.fromisoformat(item.get('timestamp', '')) < final_date]
        return filtered
    except ValueError:
        print(f"Invalid date format: {date_filter}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
        return data

def show_conversations(limit: int = 10, date_filter: tuple[str, str] = None, dev: bool = False):
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
    
    output_manager.write(f"\n{'='*50}")
    output_manager.write(f"CONVS (last {min(limit, len(conversations))}) {'[DEV]' if dev else '[PROD]'}")
    if date_filter:
        output_manager.write(f"From: {date_filter}")
    output_manager.write(f"{'='*50}\n")
    
    for conv in conversations[-limit:]:
        output_manager.write(f"T: {format_timestamp(conv.get('timestamp', 'N/A'))[:16]}")
        output_manager.write(f"W: {conv.get('wake_word', 'N/A')[:15]}")
        output_manager.write(f"Tone: {conv.get('tone', 'N/A')}")
        output_manager.write(f"Agent: {conv.get('agent_name', 'Final Answer')}")
        user_text = conv.get('user_input', conv.get('request', 'N/A'))[:40]
        output_manager.write(f"U: {user_text}")
        asst_text = conv.get('assistant_response', conv.get('response', 'N/A'))[:40]
        output_manager.write(f"A: {asst_text}")
        output_manager.write("-" * 50)

def show_token_usage_stats(date_filter: tuple[str, str] = None, dev: bool = False):
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
    
    output_manager.write(f"\n{'='*50}")
    output_manager.write(f"TOKEN USAGE {'[DEV]' if dev else '[PROD]'}")
    if date_filter:
        output_manager.write(f"From: {date_filter}")
    output_manager.write(f"{'='*50}\n")
    
    # Group by agent
    stats_by_agent = defaultdict(lambda: {
        'calls': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'input_cost': 0.0,
        'output_cost': 0.0,
    })
    
    stats_by_model = defaultdict(lambda: {
        'calls': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'input_cost': 0.0,
        'output_cost': 0.0,
        'audio_calls': 0,
        'audio_input_tokens': 0,
        'audio_output_tokens': 0,
        'audio_total_tokens': 0,
        'audio_input_cost': 0.0,
        'audio_output_cost': 0.0,
    })

    for log in token_logs:
        agent = log.get('agent_name', 'unknown')
        agent_name = agent.split('agent')[0] + 'agent'  # Normalize agent name to base form
        model_name = AGENT_CONFIG.get(agent_name, {}).get('llm_config', {}).get('model_name', 'unknown')

        stats_by_agent[agent]['calls'] += 1
        stats_by_agent[agent]['input_tokens'] += log.get('input_tokens', 0)
        stats_by_agent[agent]['output_tokens'] += log.get('output_tokens', 0)
        stats_by_agent[agent]['total_tokens'] += log.get('total_tokens', 0)
        if 'text_to_speech' in agent_name:
            stats_by_agent[agent]['input_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('input', 0) * log.get('input_tokens', 0) / 1_000_000
            stats_by_agent[agent]['output_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('audio_output', 0) * log.get('output_tokens', 0) / 1_000_000

            stats_by_model[model_name]['audio_calls'] += 1
            stats_by_model[model_name]['input_tokens'] += log.get('input_tokens', 0)
            stats_by_model[model_name]['audio_output_tokens'] += log.get('output_tokens', 0)
            stats_by_model[model_name]['total_tokens'] += log.get('total_tokens', 0)
            stats_by_model[model_name]['input_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('input', 0) * log.get('input_tokens', 0) / 1_000_000
            stats_by_model[model_name]['audio_output_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('audio_output', 0) * log.get('output_tokens', 0) / 1_000_000
        elif 'speech_to_text' in agent_name:
            stats_by_agent[agent]['input_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('audio_input', 0) * log.get('input_tokens', 0) / 1_000_000
            stats_by_agent[agent]['output_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('output', 0) * log.get('output_tokens', 0) / 1_000_000
            
            stats_by_model[model_name]['audio_calls'] += 1
            stats_by_model[model_name]['audio_input_tokens'] += log.get('input_tokens', 0)
            stats_by_model[model_name]['output_tokens'] += log.get('output_tokens', 0)
            stats_by_model[model_name]['total_tokens'] += log.get('total_tokens', 0)
            stats_by_model[model_name]['audio_input_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('audio_input', 0) * log.get('input_tokens', 0) / 1_000_000
            stats_by_model[model_name]['output_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('output', 0) * log.get('output_tokens', 0) / 1_000_000
        else:
            stats_by_agent[agent]['input_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('input', 0) * log.get('input_tokens', 0) / 1_000_000
            stats_by_agent[agent]['output_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('output', 0) * log.get('output_tokens', 0) / 1_000_000

            stats_by_model[model_name]['calls'] += 1
            stats_by_model[model_name]['input_tokens'] += log.get('input_tokens', 0)
            stats_by_model[model_name]['output_tokens'] += log.get('output_tokens', 0)
            stats_by_model[model_name]['total_tokens'] += log.get('total_tokens', 0)
            stats_by_model[model_name]['input_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('input', 0) * log.get('input_tokens', 0) / 1_000_000
            stats_by_model[model_name]['output_cost'] += COST_CONFIG.get(model_name, {}).get('cost_per_million', {}).get('output', 0) * log.get('output_tokens', 0) / 1_000_000
    
    # Display per-agent stats
    total_calls = 0
    total_input = 0
    total_output = 0
    total_tokens = 0
    total_input_cost = 0
    total_output_cost = 0
    
    for agent in sorted(stats_by_agent.keys()):
        stats = stats_by_agent[agent]
        output_manager.write(f"{agent}:")
        output_manager.write(f"  Calls: {stats['calls']}")
        output_manager.write(f"  In: {stats['input_tokens']:<4} Out: {stats['output_tokens']}")
        output_manager.write(f"  Total: {stats['total_tokens']}")
        output_manager.write(f"  Cost: {stats['input_cost']:.4f}$ (in) + {stats['output_cost']:.4f}$ (out) = {stats['input_cost'] + stats['output_cost']:.4f}$")
        
        total_calls += stats['calls']
        total_input += stats['input_tokens']
        total_output += stats['output_tokens']
        total_tokens += stats['total_tokens']
        total_input_cost += stats['input_cost']
        total_output_cost += stats['output_cost']
    
    output_manager.write("\n" + "-" * 50 + "\n")    

    for model in sorted(stats_by_model.keys()):
        stats = stats_by_model[model]
        output_manager.write(f"{model}:")
        output_manager.write(f"  Calls: {stats['calls']}")
        output_manager.write(f"  In: {stats['input_tokens']:<4} Out: {stats['output_tokens']}")
        output_manager.write(f"  Total: {stats['total_tokens']}")
        output_manager.write(f"  Cost: {stats['input_cost']:.4f}$ (in) + {stats['output_cost']:.4f}$ (out) = {stats['input_cost'] + stats['output_cost']:.4f}$")

        if stats['audio_calls'] > 0:
             output_manager.write(f"  Audio Calls: {stats['audio_calls']}")
             output_manager.write(f"  Audio In: {stats['audio_input_tokens']:<4} Out: {stats['audio_output_tokens']}")
             output_manager.write(f"  Audio Total: {stats['audio_total_tokens']}")
             output_manager.write(f"  Audio Cost: {stats['audio_input_cost']:.4f}$ (in) + {stats['audio_output_cost']:.4f}$ (out) = {stats['audio_input_cost'] + stats['audio_output_cost']:.4f}$")        

    output_manager.write("-" * 50)
    output_manager.write(f"Total: {total_input:,} (in) + {total_output:,} (out) = {total_tokens:,} tokens")
    output_manager.write(f"Total Cost: {total_input_cost:.4f}$ (in) + {total_output_cost:.4f}$ (out) = {total_input_cost + total_output_cost:.4f}$")
    output_manager.write(f"Calls: {total_calls}")
    

def show_errors(limit: int = 10, date_filter: tuple[str, str] = None, dev: bool = False):
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
    
    output_manager.write(f"\n{'='*50}")
    output_manager.write(f"ERRORS (last {min(limit, len(errors))}) {'[DEV]' if dev else '[PROD]'}")
    if date_filter:
        output_manager.write(f"From: {date_filter}")
    output_manager.write(f"{'='*50}\n")
    
    for error in errors[-limit:]:
        output_manager.write(f"T: {format_timestamp(error.get('timestamp', 'N/A'))[:16]}")
        output_manager.write(f"Type: {error.get('error_type', 'N/A')}")
        msg = error.get('error_message', 'N/A')[:40]
        output_manager.write(f"Msg: {msg}")
        output_manager.write("-" * 50)

def show_overall_stats(date_filter: tuple[str, str] = None, dev: bool = False):
    """Display overall statistics."""
    log_prefix = "dev_" if dev else ""
    conversations = load_jsonl(LOGS_DIR / f"{log_prefix}conversations.jsonl")
    token_logs = load_jsonl(LOGS_DIR / f"{log_prefix}token_usage.jsonl")
    errors = load_jsonl(LOGS_DIR / f"{log_prefix}errors.jsonl")
    
    conversations = filter_data(conversations, date_filter)
    token_logs = filter_data(token_logs, date_filter)
    errors = filter_data(errors, date_filter)
    
    output_manager.write(f"\n{'='*50}")
    output_manager.write(f"STATS {'[DEV]' if dev else '[PROD]'}")
    if date_filter:
        output_manager.write(f"From: {date_filter}")
    output_manager.write(f"{'='*50}\n")
    
    # Calculate time range
    if conversations:
        first_time = datetime.fromisoformat(conversations[0]['timestamp'])
        last_time = datetime.fromisoformat(conversations[-1]['timestamp'])
        duration = last_time - first_time
        output_manager.write(f"Conversations: {len(conversations)}")
        days = duration.days
        output_manager.write(f"Duration: {days}d")
    else:
        output_manager.write("Conversations: 0")
    
    # Token statistics
    if token_logs:
        total_tokens = sum(log.get('total_tokens', 0) for log in token_logs)
        output_manager.write(f"\nLLM Calls: {len(token_logs)}")
        output_manager.write(f"Tokens: {total_tokens:,}")
        avg_tokens = total_tokens / len(token_logs) if token_logs else 0
        output_manager.write(f"Avg: {avg_tokens:.0f}/call")
    else:
        output_manager.write("\nLLM Calls: 0")
    
    # Tone distribution
    if conversations:
        tone_counts = defaultdict(int)
        for conv in conversations:
            tone = conv.get('tone', 'unknown')
            tone_counts[tone] += 1
        output_manager.write(f"\nTones:")
        for tone, count in sorted(tone_counts.items()):
            output_manager.write(f"  {tone}: {count}")
    
    # Error statistics
    if errors:
        error_types = defaultdict(int)
        for error in errors:
            error_type = error.get('error_type', 'unknown')
            error_types[error_type] += 1
        output_manager.write(f"\nErrors: {len(errors)}")
        for error_type, count in sorted(error_types.items()):
            output_manager.write(f"  {error_type}: {count}")
    else:
        output_manager.write("\nErrors: 0")
    
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
        output_manager.write(f"Agent: {conv.get('agent_name', 'N/A')}")
        output_manager.write(f"\nUser:\n{conv.get('user_input', conv.get('request', 'N/A'))}")
        output_manager.write(f"\nAssistant:\n{conv.get('assistant_response', conv.get('response', 'N/A'))}")
        output_manager.write("\n" + "-" * 80)

def send_email(file_path: Path, content: str = None):
    """Send email with analysis file as attachment."""
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    msg['Subject'] = 'Home Assistant Token Analysis'
    
    if content is None:
        body = "Please find the attached analysis report."
    else:
        body = content
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach the file
    with open(file_path, 'rb') as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
    
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename= {file_path.name}')
    msg.attach(part)
    
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.ehlo()
    smtp_server.starttls()
    smtp_server.login(EMAIL_FROM, EMAIL_PASSWORD)
    smtp_server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    smtp_server.quit()

def send_monthly_email_report():
    global output_manager
    
    logging_month = 12 if datetime.now().month == 1 else datetime.now().month - 1
    logging_year = datetime.now().year - 1 if datetime.now().month == 1 else datetime.now().year

    with open(MONTHLY_EMAIL_PATH, 'r') as f:
        content = f.read()
    
    if f"{logging_month}-{logging_year}" in content:
        return

    initial_date = datetime(logging_year, logging_month, 1).strftime("%Y-%m-%d")
    final_date = (datetime(logging_year, logging_month, 1) + timedelta(days=32)).replace(day=1).strftime("%Y-%m-%d")
    date_filter = (initial_date, final_date)

    output_manager = OutputManager(use_file=True)
    try:
        show_conversations(10, date_filter, dev=False)
        show_token_usage_stats(date_filter, dev=False)
        show_errors(10, date_filter, dev=False)
        show_overall_stats(date_filter, dev=False)
    finally:
        output_manager.close()
    
    try:
        send_email(output_manager.file_path, content=f"Monthly Home Assistant Analysis for {initial_date} to {final_date}")
    finally:
        with open(MONTHLY_EMAIL_PATH, 'a') as f:
            f.write(f"\n{logging_month}-{logging_year}")


def main():
    global output_manager
    
    parser = argparse.ArgumentParser(description="Analyze Home Assistant logs")
    parser.add_argument("--conversations", action="store_true", help="Show recent conversations")
    parser.add_argument("--tokens", action="store_true", help="Show token usage statistics")
    parser.add_argument("--errors", action="store_true", help="Show recent errors")
    parser.add_argument("--stats", action="store_true", help="Show overall statistics")
    parser.add_argument("--conversation-id", type=str, help="Show specific conversation by ID")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of items to show (default: 10)")
    parser.add_argument("--from-date", type=str, help="Analyze data from this date onward (YYYY-MM-DD)")
    parser.add_argument("--to-date", type=str, help="Analyze data up to this date (YYYY-MM-DD)")
    parser.add_argument("--save", action="store_true", help="Save analysis to a dated file")
    parser.add_argument("--dev", action="store_true", help="Analyze dev logs instead of production logs")
    parser.add_argument("--email", action="store_true", help="Email the analysis results instead of printing")
    
    args = parser.parse_args()
    
    date_range = [args.from_date, args.to_date] if args.from_date or args.to_date else None

    # Initialize output manager
    output_manager = OutputManager(use_file=(args.save or args.email))
    
    # If no arguments provided, show all
    if not any([args.conversations, args.tokens, args.errors, args.stats, args.conversation_id]):
        args.stats = True
        args.conversations = True
        args.tokens = True
    
    try:
        if args.conversations:
            show_conversations(args.limit, date_range, args.dev)
        
        if args.tokens:
            show_token_usage_stats(date_range, args.dev)
        
        if args.errors:
            show_errors(args.limit, date_range, args.dev)
        
        if args.stats:
            show_overall_stats(date_range, args.dev)
        
        if args.conversation_id:
            show_conversation_by_id(args.conversation_id, args.dev)

    finally:
        output_manager.close()
    
    # Send email after file is closed
    if args.email:
        send_email(output_manager.file_path)
        print(f"Analysis emailed to {EMAIL_TO}")

if __name__ == "__main__":
    main()