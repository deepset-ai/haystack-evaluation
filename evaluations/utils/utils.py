import time
from datetime import datetime
from functools import wraps


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        formatted_time = str(datetime.utcfromtimestamp(total_time).strftime("%H:%M:%S"))
        print(f"Function {func.__name__} took {formatted_time} ({total_time:.4f} seconds)")
        return result

    return timeit_wrapper
