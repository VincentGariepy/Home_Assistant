import datetime
import wave
import pyaudio
import time

from langchain_core.tools import tool

from audio_functions.genai_audio_utils import setup_audio_stream

class Timers():
    def __init__(self):
        self.timers = {}
    
    def create_timer(self, name, total_time):
        if name in self.timers:
            raise ValueError(f"Timer with name '{name}' already exists.")
        self.timers[name] = Timer(name, total_time)
    
    def get_timer(self, name):
        if name not in self.timers:
            raise ValueError(f"Timer with name '{name}' does not exist.")
        return self.timers[name]
    
    def list_timers(self):
        return list(self.timers.keys())
    
    def timer_data(self):
        timer_data = {}
        for name, timer in self.timers.items():
            timer_data[name] = {
                'elapsed_time': timer.get_elapsed_time(),
                'remaining_time': timer.get_remaining_time()
             }
        return timer_data
    
    def delete_timer(self, name):
        if name not in self.timers:
            raise ValueError(f"Timer with name '{name}' does not exist.")
        del self.timers[name]

    def check_timers(self):
        if not self.timers:
            return []
        timer_names = []
        for timer in self.timers.values():
            if timer.check_timer():
                timer_names.append(timer.get_name())
        for timer_name in timer_names:        
            self.delete_timer(timer.get_name())
        return timer_names

class Timer():
    def __init__(self, name, total_time):
        self.name = name
        self.total_time = total_time
        self.start_time = datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(seconds=total_time)
        self._validate_time()

    def _validate_time(self):
        if self.total_time > 86400:
            raise ValueError("Total time cannot exceed 24 hours (86400 seconds).")

    def get_name(self):
        return self.name
    
    def get_elapsed_time(self):
        time_elapse = datetime.datetime.now() - self.start_time
        return {'hours': time_elapse.seconds // 3600, 'minutes': (time_elapse.seconds % 3600) // 60, 'seconds': time_elapse.seconds % 60}
    
    def get_remaining_time(self):
        time_remaining = self.end_time - datetime.datetime.now()
        return {'hours': time_remaining.seconds // 3600, 'minutes': (time_remaining.seconds % 3600) // 60, 'seconds': time_remaining.seconds % 60}
    
    def _ring(self):
        file_path = r"./tools/rooster_alarm.wav"
        try:
            f = wave.open(file_path, "rb")
            
            # Get WAV file properties
            sampwidth = f.getsampwidth()
            channels = f.getnchannels()
            framerate = f.getframerate()
            
            # Convert sampwidth to PyAudio format
            if sampwidth == 1:
                audio_format = pyaudio.paUInt8
            elif sampwidth == 2:
                audio_format = pyaudio.paInt16
            elif sampwidth == 4:
                audio_format = pyaudio.paInt32
            else:
                audio_format = pyaudio.paInt16  # Default fallback
            
            # Try the original sample rate first; if it fails, try common alternatives
            sample_rates_to_try = [framerate, 44100, 48000]
            
            stream = None
            p = None
            for rate in sample_rates_to_try:
                try:
                    stream, p = setup_audio_stream(audio_format, channels, rate)
                    break
                except OSError:
                    if stream:
                        stream.close()
                    if p:
                        p.terminate()
                    continue
            
            if stream is None:
                raise RuntimeError("Could not open audio stream with any supported sample rate")
            
            chunk = 1024
            data = f.readframes(chunk)

            for _ in range(2):  # Ring the alarm 2 times
                while len(data) > 0:
                    stream.write(data)
                    data = f.readframes(chunk)
                time.sleep(2)
                f.rewind()
                data = f.readframes(chunk)

            stream.stop_stream()
            stream.close()
            p.terminate()
            f.close()
        except Exception as e:
            print(f"Failed to play alarm: {e}")

    def check_timer(self):
        if datetime.datetime.now() >= self.end_time:
            print("Ringing .. ")
            self._ring()
            return True
        return False
    
timers_manager = Timers()

@tool
def add_timer(name: str, total_time: int) -> None:
    """
    Add a timer with the given name and total time in seconds.
    
    Args:
        name (str): The name of the timer.
        total_time (int): The total time for the timer in seconds (max 86400 seconds or 24 hours).
    """
    timers_manager.create_timer(name, total_time)

@tool
def timer_data() -> dict:
    """
    Return a dictionary with timer names as keys and their elapsed and remaining time as values.
    
    Returns:
        dict: A dictionary where keys are timer names and values are dictionaries with 'elapsed_time' and 'remaining_time'.
    """
    return timers_manager.timer_data()

def check_timers() -> list:
    """
    Checks all timers and `rings` the ones that have finished. Then return a list of timer names that have completed.
    
    Returns:
        list: A list of timer names that have completed.
    """
    return timers_manager.check_timers()

@tool
def convert_minutes_to_seconds(minutes: int) -> int:
    """
    Convert minutes to seconds.
    
    Args:
        minutes (int): The number of minutes to convert.

    Returns:
        int: The equivalent number of seconds.
    """
    return minutes * 60

@tool
def convert_hours_to_seconds(hours: int) -> int:
    """
    Convert hours to seconds.
    
    Args:
        hours (int): The number of hours to convert.
    
    Returns:
        int: The equivalent number of seconds.
    """
    return hours * 3600