from google import genai
from google.genai import types
import wave
import os

import yaml

# Set up the wave file to save the output:
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
   with wave.open(filename, "wb") as wf:
      wf.setnchannels(channels)
      wf.setsampwidth(sample_width)
      wf.setframerate(rate)
      wf.writeframes(pcm)

def convert_text_to_speech(content: str, file_name:str, tone: str):
    ai_api_key = os.getenv('GOOGLE_AI_KEY')
    with open('agents/agents_config.yml', 'r') as file:
        config = yaml.safe_load(file)
    
    agent_config = config['text_to_speech_agent']['llm_config']

    if tone == 'sassy':
        llm_config = agent_config['sassy']
    elif tone == 'scottish':
        llm_config = agent_config['scottish']
    else:
        llm_config = agent_config['neutral']  

    client = genai.Client(api_key=ai_api_key)

    content = llm_config['system_message'].format(content)

    response = client.models.generate_content(
        model=agent_config['model_name'],
        contents=content,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=llm_config['voice_name'],
                    )
                )
            ),
        )
    )

    data = response.candidates[0].content.parts[0].inline_data.data

    wave_file(file_name, data) # Saves the file to current directory