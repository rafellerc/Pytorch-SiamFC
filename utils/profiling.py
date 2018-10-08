import time


# TODO have a option that converts time into minutes and hours as needed.
class Timer(object):
    def __init__(self, print_at_exit=False, convert=False):
        self.print_at_exit = print_at_exit
        self.convert = convert
        self.exited = False
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        if self.print_at_exit:
            print(self)

    def reset(self):
        self.start_time = time.time()

    def __repr__(self):
        return "<{} elapsed={}>".format(self.__class__.__name__,
                                        self.elapsed)

    @property
    def elapsed(self):
        # If called inside the context manager it calculates the partial elapsed
        # time as well.
        if not self.exited:
            self.end_time = time.time()
        elap = self.end_time - self.start_time
        if self.convert:
            sec = (elap) % 60
            minut = (elap // 60) % 60
            hour = elap // 3600
            return "{:.0f}h{:.0f}m{:.2f}s".format(hour, minut, sec)
        else:
            return elap


def _main():
    with Timer() as tim:
        for i in range(3):
            print(tim.elapsed)
            time.sleep(1)
            for j in range(3):
                print(tim.elapsed)
                time.sleep(0.5)


if __name__ == "__main__":
    _main()
