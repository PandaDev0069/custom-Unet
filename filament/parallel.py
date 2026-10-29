"""Bounded, ordered CPU work without queuing native-resolution arrays in bulk."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def ordered_map(function, items, workers=0):
    """At most `workers` pending jobs; propagate failures and preserve input order.

    OpenCV, NumPy, SciPy and COCO do their expensive work in native code. Threads
    share compact metadata and avoid copying full-resolution masks between processes.
    The serial path is useful for debugging and small synthetic checks.
    """
    if workers < 0:
        raise ValueError("workers cannot be negative")
    if workers <= 1:
        yield from map(function, items)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="filament-cpu") as pool:
        pending = deque()
        for _ in range(workers):
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            result = pending.popleft().result()
            yield result
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                pass
