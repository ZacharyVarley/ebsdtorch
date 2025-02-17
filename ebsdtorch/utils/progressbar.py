"""

Not having tqdm as a dependency so I wrote a simple progress bar.

"""

import sys
import time


def progressbar(it, prefix="", prefix_min_length=15, size=60, out=sys.stdout):
    """
    Progress bar for iterators.

    Args:
        it (Iterable): The iterator to wrap.
        prefix (str): The prefix to print before the progress bar.
        prefix_min_lenght (int): The minimum length of the prefix.
        size (int): The size of the progress bar in characters.
        out (file): Write destination.

    """
    prefix = "{:<{}}".format(prefix, prefix_min_length)
    count = len(it)
    start = time.time()

    for i, item in enumerate(it):
        yield item
        j = i + 1
        x = int(size * j / count)
        seconds_per_iteration = (time.time() - start) / j
        remaining = (count - j) * seconds_per_iteration

        # remaining
        r_mins, r_sec = divmod(remaining, 60)
        r_time_str = (
            f"{int(r_mins):02}:{r_sec:04.1f}" if r_mins > 0 else f"{r_sec:04.1f}"
        )

        # elapsed
        e_mins, e_sec = divmod(time.time() - start, 60)
        e_time_str = (
            f"{int(e_mins):02}:{e_sec:04.1f}" if e_mins > 0 else f"{e_sec:04.1f}"
        )

        # current iterations per second
        ips = 1.0 / (seconds_per_iteration + 1e-8)
        speed_str = f"{ips:04.1f} itr/s" if ips > 1.0 else f"{1.0/ips:04.1f} s/itr"

        print(
            f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j / count * 100:04.1f}% {j}/{count} {e_time_str}<{r_time_str}, {speed_str}",
            end="\r",
            file=out,
            flush=True,
        )
    print("\n")
