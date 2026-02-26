import os
import io
import wave
import time
import numpy as np
import pyaudio
import openwakeword
from openwakeword.model import Model

# To train onnx models use this notebook : https://colab.research.google.com/drive/1zzKpSnqVkUDD3FyZ-Yxw3grF7L0R1rlk#scrollTo=step4_train
# An alternative is to use this free API : https://console.picovoice.ai/signup

WAKE_WORDS = ['hey_jarvis', 'models/wake_words/a_yi_la_poot.onnx', 'models/wake_words/oiy_mate.onnx'] 
CHANNELS = 1
RATE = 16000
CHUNK = 1280  # openWakeWord prefers 1280 sample chunks (80ms)

def _get_input_device_index(p):
    """
    Helper to find the correct audio input device.
    Useful for Raspberry Pi where 'default' might be HDMI.
    """
    target_index = None
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        # Filter for devices that actually have input channels
        if info['maxInputChannels'] > 0:
            name = info['name']
            
            # Priority 1: Look for USB devices (most common for Pi mics)
            if "USB" in name:
                return i
            
            # Priority 2: Look for specific hardware like ReSpeaker or common HATs
            if any(key in name for key in ["mic", "shure", "akg", "re-speaker"]):
                target_index = i

    # If we found a specific USB device, use it. Otherwise, let PyAudio choose default.
    return target_index

def _record_user_request(stream):
    """Records audio after wake word until silence."""
    print("Listening for request...")
    frames = []
    silence_threshold = 500  # Adjust based on mic sensitivity
    silent_chunks = 0
    max_silence = 20 # ~2 seconds of silence (20 * 0.1s chunks)

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Simple volume check for silence detection
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.abs(audio_data).max() < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
            
        if silent_chunks > max_silence:
            break
            
    # Wrap in RAM as a WAV file
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    buffer.seek(0)
    return buffer

def get_user_request():
    # Initialize openWakeWord with multiple models
    oww_model = Model(wakeword_models=WAKE_WORDS, inference_framework="onnx")

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK, input_device_index=_get_input_device_index(pa))

    print(f"System Offline. Monitoring for: {WAKE_WORDS}")

    try:
        while True:
            # Read mic data
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_numpy = np.frombuffer(data, dtype=np.int16)

            # Process with openWakeWord
            # Returns a dict: {'wakeword_name': confidence_score}
            prediction = oww_model.predict(audio_numpy)
            
            wake_word = max(prediction.items(), key=lambda x: x[1])[0]

            for word, score in prediction.items():
                if score > 0.5:  # Threshold for detection
                    print(f"\nTrigger Detected: {word} (Score: {score:.2f})")
                    
                    # Record the follow-up request
                    audio_file = _record_user_request(stream)
                    return audio_file, wake_word
                    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()