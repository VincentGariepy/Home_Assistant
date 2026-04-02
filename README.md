# Home Assistant - Agentic Voice-Controlled Assistant

A sophisticated, multi-agent home assistant system powered by Google's Gemini AI that provides voice-controlled access to weather, navigation, online search, timers, and code execution. The system features multiple response tones (neutral, sassy, scottish) and intelligent agent routing.

## Features

- **Voice Interface**: Wake word detection with multiple personality modes
- **Multi-Agent Architecture**: Specialized agents for different tasks with LangGraph integration
- **Location Services**: Real-time weather (current, hourly, daily) and route navigation
- **Online Search**: Web search capabilities for up-to-date information
- **Timer Management**: Set and manage multiple timers with voice control
- **Code Execution**: Execute Python code to answer computational queries
- **Personality Modes**: Response styles including neutral, sassy, and Scottish
- **Comprehensive Logging**: Conversation logging, token usage tracking, and error monitoring
- **Cost Analysis**: Monthly reports on model usage and API costs
- **Text-to-Speech**: Multiple voice options matching response tones

## System Architecture

### Agent Structure

The system uses a hierarchical multi-agent architecture:

1. **Main Agent** - Orchestrator
   - Routes user queries to specialized agents
   - Responds to general knowledge questions
   - Adapts tone based on wake word (neutral, sassy, scottish)

2. **Specialized Agents**
   - **Route Agent**: Google Maps integration for navigation and traffic info
   - **Weather Agent**: Real-time weather queries and forecasts
   - **Online Search Agent**: Web searches for current information
   - **Code Agent**: Executes Python code for computational tasks
   - **Timer Agent**: Manages timers and scheduling

3. **Support Agents**
   - **Quirky Agent**: Provides entertaining messages while main agent processes requests
   - **Text-to-Speech Agent**: Converts responses to speech with tone-specific voices
   - **Speech-to-Text Agent**: Transcribes audio input

### Key Technologies

- **LangChain & LangGraph**: Agentic framework and tool calling
- **Google Gemini AI**: Primary LLM (gemini-3-flash-preview)
- **OpenWakeWord**: Multi-model wake word detection
- **Google APIs**: Maps, Weather, Search, Speech, Text-to-Speech
- **PyAudio**: Audio input/output processing
- **JSONL Logging**: Structured conversation and token usage logs

## Project Structure

```
Home_Assistant/
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── Ideas.md                         # Future feature ideas
│
├── agents/                          # Agent system
│   ├── agents.py                    # Agent implementations and tool calling logic
│   ├── home_assistant.py            # HomeAssistant main class and orchestration
│   ├── agent_init_data.py           # Agent data structure
│   ├── agents_config.yaml           # Agent configurations and prompts
│   └── utils.py                     # Agent utilities
│
├── audio_functions/                 # Audio processing
│   ├── audio_listening.py           # Wake word detection and recording
│   ├── genai_audio_utils.py         # Text-to-speech and speech-to-text
│   └── __init__.py
│
├── tools/                           # Specialized tools for agents
│   ├── weather_tools.py             # Weather API integrations
│   ├── places_tools.py              # Google Maps integration
│   ├── timer_tool.py                # Timer management
│   ├── utils.py                     # Tool utilities (predefined addresses)
│   └── __init__.py
│
├── log_config/                      # Logging and monitoring
│   ├── logging_config.py            # Logging setup
│   ├── log_analyzer.py              # Monthly cost analysis and reporting
│   ├── model_costs.yaml             # API pricing configuration
│   ├── monthly_email.txt            # Email template for reports
│   ├── analysis/                    # Monthly analysis reports
│   └── __init__.py
│
├── logs/                            # Runtime logs
│   ├── conversations.jsonl          # Production conversations
│   ├── dev_conversations.jsonl      # Development mode conversations
│   ├── errors.jsonl                 # Error logs
│   ├── dev_errors.jsonl             # Development mode errors
│   ├── token_usage.jsonl            # Token usage tracking
│   └── dev_token_usage.jsonl        # Development mode token usage
│
└── models/
    └── wake_words/                  # Custom ONNX wake word models
        ├── a_yi_la_poot.onnx        # Sassy tone trigger
        └── oiy_mate.onnx            # Scottish tone trigger
```

## Installation

### Prerequisites

- Python 3.10+
- Google Cloud API credentials (Maps, Weather, Search, Gemini, Speech, Text-to-Speech)
- Audio input device (microphone)
- Audio output device (speakers)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Home_Assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   export GOOGLE_AI_KEY="your_gemini_api_key"
   export GOOGLE_MAPS_KEY="your_maps_api_key"
   export GOOGLE_CSE_ID="your_search_engine_id"
   export GMAIL_VGARIEPY339_APP_PASSWORD="your_gmail_app_password"  # For monthly reports
   ```

5. **Configure predefined locations**
   Edit `tools/utils.py` to add your custom addresses:
   ```python
   ADDRESSES = {
       'home_address': 'Your Home Address, City, Country',
       'work_address': 'Your Work Address, City, Country',
       # Add more locations as needed
   }
   ```

## Usage

### Basic Startup

```bash
# Run in production mode with audio
python main.py

# Run in development mode (logging tokens, conversations, etc. to dev logs)
python main.py --dev

# Run with verbose debugging
python main.py --debug

# Run without audio interface
python main.py --no_audio
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dev` | Run in development mode with dev logging | False |
| `--debug` | Run in debug mode with verbose logging | False |
| `--memory_timeframe` | Conversation memory timeframe in minutes (0-120) | 5 |
| `--no_audio` | Run without audio input/output | False |
| `--quirky_response_interval` | Minutes between quirky responses (-1=never, 3, 5, 10, 15) | 10 |

### Example Commands

```bash
# Run with 10-minute conversation memory and sassy responses every 5 minutes
python main.py --quirky_response_interval 5 --memory_timeframe 10

# Run in development mode with no audio and 20-minute memory
python main.py --dev --no_audio --memory_timeframe 20
```

## Wake Words & Personality Modes

### Wake Word Detection

The system monitors for three wake words, each triggering a different response personality:

1. **"Hey Jarvis"** (Built-in OpenWakeWord model)
   - Response Tone: Neutral
   - Voice: Autonoe

2. **Custom Wake Word 1** (`models/wake_words/a_yi_la_poot.onnx`)
   - Response Tone: Sassy
   - Voice: Laomedeia

3. **Custom Wake Word 2** (`models/wake_words/oiy_mate.onnx`)
   - Response Tone: Scottish
   - Voice: Iapetus

### Training Custom Wake Words

To create custom wake word models, use this [OpenWakeWord training notebook](https://colab.research.google.com/drive/1zzKpSnqVkUDD3FyZ-Yxw3grF7L0R1rlk#scrollTo=step4_train).

Alternatively, use [Picovoice Console](https://console.picovoice.ai/signup) for an api option.

## Agent Details

### Main Agent

The orchestrator that:
- Understands natural language queries
- Routes to specialized agents when needed
- Answers from general knowledge when applicable
- Adapts responses based on personality tone

**Model**: gemini-3-flash-preview  
**Temperature**: 0 (deterministic)

### Route Agent

Handles navigation queries using Google Maps API.

**Capabilities**:
- Get driving routes and directions
- Check traffic conditions
- Calculate travel time
- Access predefined location addresses

**Model**: gemini-3-flash-preview

### Weather Agent

Provides weather information through Google Weather API.

**Capabilities**:
- Current weather conditions
- Daily forecasts (up to 10 days)
- Hourly forecasts
- Location lookup support

**Model**: gemini-3-flash-preview

### Online Search Agent

Performs web searches for current information.

**Capabilities**:
- General knowledge queries
- Current events and news
- Web page search
- Time-aware results

**Model**: gemini-3-flash-preview

### Code Agent

Executes Python code to answer computational questions.

**Capabilities**:
- Mathematical computations
- Random operations
- Data processing
- Logic operations

**Model**: gemini-3-flash-preview

### Timer Agent

Manages timers and scheduling.

**Capabilities**:
- Add timers with custom names
- List active timers
- Check elapsed/remaining time
- Support for hours, minutes, seconds

**Model**: gemini-3-flash-preview

### Quirky Agent

Provides entertaining placeholder responses while the main agent processes.

**Capabilities**:
- Short, fun messages (≤5 words)
- Tone-matched responses (neutral, sassy, scottish)
- Personable messaging ("I'm...")

**Model**: gemini-3.1-flash-lite-preview  
**Temperature**: 2.0 (creative/random)

## Configuration

### Agent Configuration

Edit `agents/agents_config.yaml` to modify:
- Agent system prompts
- Model selection (e.g., gemini-3-flash vs gemini-3.1-flash-lite)
- Temperature settings (creativity/randomness)
- Tool descriptions and behavior

### Logging Configuration

Configure in `log_config/logging_config.py`:
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Output destinations (console, files)
- Conversation history storage
- Development vs production modes

### Cost Tracking

Update `log_config/model_costs.yaml` with current API pricing to track costs accurately in monthly reports.

## Logging & Monitoring

### Log Files

Located in `logs/` directory:

- **conversations.jsonl**: Production conversation records
  - User input, assistant response, wake word, tone
  - Token usage and model information

- **errors.jsonl**: Production error logs
  - Error type, message, context, timestamp

- **token_usage.jsonl**: API token tracking
  - Model name, input/output tokens, reasoning tokens
  - Cost calculation

### Monthly Reports

The system automatically generates monthly cost analysis reports:
- API token usage per model
- Estimated costs based on pricing
- Usage patterns and trends
- Sent via email (requires Gmail setup)

To send reports, ensure `GMAIL_VGARIEPY339_APP_PASSWORD` is set.

## Dependencies

Key Python packages:
- `langgraph` - Agent framework
- `langchain` - LLM chains and tools
- `langchain-google-genai` - Gemini AI integration
- `google-cloud-*` - Google Cloud services
- `pyaudio` - Audio I/O
- `openwakeword` - Wake word detection
- `pydantic` - Data validation
- `pyyaml` - Configuration files
- `requests` - HTTP requests
- `google-auth` - API authentication

See `requirements.txt` for complete list with versions.

## Future Enhancements

See `Ideas.md` for planned features including:
- Physical device integration (buttons, LEDs)
- Health monitoring and auto-restart
- Advanced audio processing
- Device control capabilities
