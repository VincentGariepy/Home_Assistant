import os
import io
import wave
import numpy as np
import pyaudio
import openwakeword
from openwakeword.model import Model

from audio_functions.genai_audio_utils import ignore_alsa_warnings
from log_config.logging_config import logger

# To train onnx models use this notebook : https://colab.research.google.com/drive/1zzKpSnqVkUDD3FyZ-Yxw3grF7L0R1rlk#scrollTo=step4_train
# An alternative is to use this free API : https://console.picovoice.ai/signup
WAKE_WORDS = [
    openwakeword.models['hey_jarvis']["model_path"], 
    'models/wake_words/heille_la_pute.onnx', 
    'models/wake_words/oi_mate.onnx',
    'models/wake_words/hey_babe.onnx',
    'models/wake_words/jarjar.onnx',
] 
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
    logger.info("Listening for request...")
    frames = []
    prev_volume = 0
    silence_threshold = 0.5
    silent_chunks = 0
    max_silence = 40 # ~2 seconds of silence (20 * 0.1s chunks)
    min_frames = 20  # Ensure at least 2 seconds of recording before detecting silence

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Simple volume check for silence detection
        audio_data = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_data).max()
        
        if prev_volume == 0:
            prev_volume = volume
            silent_chunks = 0
        elif volume * silence_threshold < prev_volume < volume * 1/silence_threshold:
            silent_chunks += 1
            prev_volume = volume
        else:   
            silent_chunks = 0
            prev_volume = volume

        # Only exit if we have enough data AND detected silence
        if len(frames) > min_frames and silent_chunks > max_silence:
            logger.info("Request recorded.")
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
    with ignore_alsa_warnings():
        oww_model = Model(wakeword_model_paths=WAKE_WORDS)
        pa = pyaudio.PyAudio()

    stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK, input_device_index=_get_input_device_index(pa))

    logger.info(f"System Online. Monitoring for: {[os.path.basename(path)[0:-5] for path in WAKE_WORDS]}")

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
                    logger.info(f"\nTrigger Detected: {word} (Score: {score:.2f})")
                    
                    # Record the follow-up request
                    audio_file = _record_user_request(stream)
                    return audio_file, wake_word
                    
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        