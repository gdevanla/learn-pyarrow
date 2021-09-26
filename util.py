import datetime
import functools
import time

# Taken from ISLib


class Timer:
    """A simple object for timing."""

    def __init__(self):
        """Initializing always starts the timer."""
        self.start()

    def start(self):
        """Explicit start method; will clear previous values. Start always happens on initialization."""
        self._start = time.time()
        self._stop = None
        self._past_stops = [self._start]

    def stop(self):
        self._stop = time.time()
        self._past_stops.append(self._stop)

    def clear(self):
        self._stop = None
        self._start = None

    def __call__(self):
        stop = self._stop if self._stop is not None else time.time()
        self._past_stops.append(stop)
        return stop - self._start

    def __str__(self):
        """Reports current time or, if stopped, stopped time."""
        duration = self.__call__()
        return str(datetime.timedelta(seconds=duration))
        # return str(round(duration, 4))

    def delta(self):
        """Return delta from previous calls or __str__ calls. Does not add an additional stop."""
        if len(self._past_stops) > 1:
            return self._past_stops[-1] - self._past_stops[-2]


def run_timer(f, callback=None):
    """Decorator that times and reports time of a function call"""

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        print(
            "run_timer: <start>",
            f.__name__,
        )

        t = Timer()

        post = f(*args, **kwargs)

        end_time = str(t)
        callback(t.delta())
        print("run_timer:  <stop>", f.__name__, end_time)

        return post

    return wrapped
