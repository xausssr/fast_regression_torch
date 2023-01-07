import time
from functools import wraps


def profile(operation: str):
    def time_benchmark(func):
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            args[0].benchmarks[operation] = total_time
            return result
        return timeit_wrapper
    return time_benchmark
