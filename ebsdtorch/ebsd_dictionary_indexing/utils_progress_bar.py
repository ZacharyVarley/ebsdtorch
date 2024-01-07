import sys
import time


def progressbar(it, prefix="", prefix_min_lenght=25, size=60, out=sys.stdout):
    prefix = "{:<{}}".format(prefix, prefix_min_lenght)
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
            f"{int(r_mins):02}:{r_sec:05.2f}" if r_mins > 0 else f"{r_sec:05.2f}"
        )

        # elapsed
        e_mins, e_sec = divmod(time.time() - start, 60)
        e_time_str = (
            f"{int(e_mins):02}:{e_sec:05.2f}" if e_mins > 0 else f"{e_sec:05.2f}"
        )

        # current iterations per second
        ips = 1.0 / seconds_per_iteration
        speed_str = f"{ips:05.2f} itr/sec" if ips > 1.0 else f"{1.0/ips:05.2f} sec/itr"

        print(
            f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j / count * 100:05.2f}% {j}/{count} {e_time_str}<{r_time_str}, {speed_str}",
            end="\r",
            file=out,
            flush=True,
        )
    print("\n")
