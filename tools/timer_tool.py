import asyncio
import datetime
import wave

from distro import name

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
                self.delete_timer(name)
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

            stream, p = setup_audio_stream(f.getsampwidth(), f.getnchannels(), f.getframerate())
            chunk = 1024

            data = f.readframes(chunk)

            for _ in range(3):  # Ring the alarm 3 times
                while len(data) > 0:
                    stream.write(data)
                    data = f.readframes(chunk)
                datetime.time.sleep(2)
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
            self._ring()
            return True
        return False
    
timers_manager = Timers()

def add_timer(name: str, total_time: int) -> None:
    """Add a timer with the given name and total time in seconds."""
    timers_manager.create_timer(name, total_time)

def timer_data() -> dict:
    """Return a dictionary with timer names as keys and their elapsed and remaining time as values."""
    return timers_manager.timer_data()

def check_timers() -> list:
    """Return a list of timer names that have completed."""
    return timers_manager.check_timers()

def convert_minutes_to_seconds(minutes: int) -> int:
    """Convert minutes to seconds."""
    return minutes * 60

def convert_hours_to_seconds(hours: int) -> int:
    """Convert hours to seconds."""
    return hours * 3600