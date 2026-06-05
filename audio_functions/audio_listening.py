import os
import numpy as np
import pyaudio
import openwakeword
from openwakeword.model import Model

from audio_functions.genai_audio_utils import ignore_alsa_warnings
from log_config.logging_config import logger

# To train onnx models use this notebook : https://colab.research.google.com/drive/1zzKpSnqVkUDD3FyZ-Yxw3grF7L0R1rlk#scrollTo=step4_train
# An alternative is to use this free API : https://console.picovoice.ai/signup
WAKE_WORDS = {
    'hey_jarvis_v0.1': {'model_path': openwakeword.models['hey_jarvis']["model_path"], 'threshold': 0.4},
    'heille_la_pute': {'model_path': 'models/wake_words/heille_la_pute.onnx', 'threshold': 0.0025},
    'oi_mate': {'model_path': 'models/wake_words/oi_mate.onnx', 'threshold': 0.0010},
    'hey_babe': {'model_path': 'models/wake_words/hey_babe.onnx', 'threshold': 0.0025},
    'jarjar': {'model_path': 'models/wake_words/jarjar.onnx', 'threshold': 0.0025},
}
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

def get_wake_word():
    with ignore_alsa_warnings():
        oww_model = Model(wakeword_model_paths=[w['model_path'] for w in WAKE_WORDS.values()])
        pa = pyaudio.PyAudio()

    stream = pa.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK, input_device_index=_get_input_device_index(pa))

    logger.info(f"System Online. Monitoring for: {', '.join(WAKE_WORDS.keys())}")

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_numpy = np.frombuffer(data, dtype=np.int16)

            prediction = oww_model.predict(audio_numpy)
            print(prediction['oi_mate'], end='\r')  # Print the score for one wake word for monitoring
            wake_word = max(prediction.items(), key=lambda x: x[1])[0]

            if prediction[wake_word] > WAKE_WORDS[wake_word]['threshold']:
                logger.info(f"\nTrigger Detected: {wake_word} (Score: {prediction[wake_word]:.6f})")
                return wake_word
                    
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()