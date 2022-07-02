__all__ = [
    "stop_threads",
    "Channel",
    "transform",
    "source",
    "sink",
    "switched",
    "TwoWayStream",
    "OneWayStream",
]

import threading


def DEBUG(mess):
    print("DEBUG %s" % mess)
    pass


# ----( threading )------------------------------------------------------------

__threads_alive = True
__conditions = []
__events = []


def threads_alive():
    global __threads_alive
    return __threads_alive


def new_thread(fun):
    threading.Thread(target=fun).start()


def new_condition():
    cond = threading.Condition()
    global __conditions
    __conditions.append(cond)
    return cond


def new_event():
    event = threading.Event()
    global __events
    __events.append(event)
    return event


def stop_threads():
    DEBUG("stopping all threads...")

    global __threads_alive
    if not __threads_alive:
        DEBUG("...threads already stopped")
        return
    __threads_alive = False

    global __conditions
    for cond in __conditions:
        cond.acquire()
        cond.notifyAll()
        cond.release()

    global __events
    for event in __events:
        event.set()

    DEBUG("...done stopping threads")


class Toggle:
    def __init__(self, start_state=False):
        self.__state = start_state
        self.__cond = new_condition()

    def wait(self, state):
        self.__cond.acquire()
        while threads_alive() and self.__state != state:
            self.__cond.wait()
        self.__cond.release()

    def toggle(self):
        self.__cond.acquire()
        self.__state = not self.__state
        self.__cond.notifyAll()
        self.__cond.release()


def debug_threads(fun, delay=5):

    import sys, traceback, time

    new_thread(fun())

    time.sleep(delay)

    print(
        "\n+------------------------------------+"
        + "\n| Press any key to see thread stacks |"
        + "\n+------------------------------------+",
        end=" ",
    )
    input()

    frames = sys._current_frames()
    for thread, frame in frames.items():
        print("\nThread %i traceback:" % thread)
        traceback.print_stack(frame)


# ----( condition patterns )----


class Channel:
    """Asynchronous data channel = producer-consumer buffer of size 1.
    The begin and end methods must be called in pairs."""

    WRITING = True
    READING = False
    __all_channels = []

    def __init__(self, data=None, writing=True, allocator=None):
        self.__data = data
        self.__allocator = allocator
        self.__toggle = Toggle(writing)
        self.__num_readers = 0
        self.__num_writers = 0
        Channel.__all_channels.append(self)

    def __alloc(self, size):
        if self.__data is None:
            print("allocating channel for %s(%i)" % (self.__allocator.__name__, size))
            self.__data = self.__allocator(size)
        else:
            assert size == len(
                self.__data
            ), "Channel size mismatch: %s new != %s old" % (size, len(self.__data))

    # validation
    def __reading(self):
        self.__num_readers += 1

    def __writing(self):
        self.__num_writers += 1

    def __validate(self):
        assert self.data is not None, "Channel data has not been allocated"
        assert self.__num_readers == 1, "Channel has no readers"
        assert self.__num_writers == 1, "Channel has no writers"

    @staticmethod
    def validate():
        for c in Channel.__all_channels:
            c.__validate()

    # These methods allow uniform treatment of Channels and tuples-of-Channels.

    @staticmethod
    def reading(arg, shape=None):
        if shape:
            if isinstance(arg, Channel):
                arg.__alloc(shape)
                arg.__reading()
            else:
                for a, s in zip(arg, shape):
                    a.__alloc(s)
                    a.__reading()
        else:
            if isinstance(arg, Channel):
                arg.__reading()
            else:
                for a in arg:
                    a.__reading()

    @staticmethod
    def writing(arg, shape=None):
        if shape:
            if isinstance(arg, Channel):
                arg.__alloc(shape)
                arg.__writing()
            else:
                for a, s in zip(arg, shape):
                    a.__alloc(s)
                    a.__writing()
        else:
            if isinstance(arg, Channel):
                arg.__writing()
            else:
                for a in arg:
                    a.__writing()

    @staticmethod
    def begin(arg, state):
        if isinstance(arg, Channel):
            arg.__toggle.wait(state)
        else:
            for a in arg:
                a.__toggle.wait(state)

    @staticmethod
    def end(arg, state):
        if isinstance(arg, Channel):
            arg.__toggle.toggle()
        else:
            for a in arg:
                a.__toggle.toggle()

    @staticmethod
    def data(arg):
        "extracts data from channel or tuple-of-channels"
        if isinstance(arg, Channel):
            return arg.__data
        else:
            return tuple(a.__data for a in arg)


# ----( decorators )----


def simple_decorator(decorator):
    "makes a decorator preserve name, doc, and dict"

    def new_decorator(fun):
        new_fun = decorator(fun)
        new_fun.__name__ = fun.__name__
        new_fun.__doc__ = fun.__doc__
        # new_fun.__dict__.update(fun.__dict__)
        return new_fun

    new_decorator.__name__ = decorator.__name__
    new_decorator.__doc__ = decorator.__doc__
    # new_decorator.__dict__.update(decorator.__dict__)

    return new_decorator


@simple_decorator
def transform(fun):
    "decorator for streaming transformations in separate threads"

    def new_fun(self, i, o):
        try:
            i_shape, o_shape = fun(self)
        except TypeError:
            i_shape, o_shape = None, None
        DEBUG("declaring transform")
        Channel.reading(i, i_shape)
        Channel.writing(o, o_shape)

        def thread_loop():
            DEBUG("starting transform")
            try:
                while threads_alive():
                    Channel.begin(i, Channel.READING)
                    Channel.begin(o, Channel.WRITING)
                    if threads_alive():
                        fun(self, Channel.data(i), Channel.data(o))
                    Channel.end(i, Channel.READING)
                    Channel.end(o, Channel.WRITING)
            except Exception as e:
                stop_threads()
                raise e
            finally:
                DEBUG("stopping transform")

        new_thread(thread_loop)

    return new_fun


@simple_decorator
def source(fun):
    "decorator for streaming data sources in separate threads"

    def new_fun(self, o):
        try:
            o_shape = fun(self)
        except TypeError:
            o_shape = None
        DEBUG("declaring source")
        Channel.writing(o, o_shape)

        def thread_loop():
            DEBUG("starting source")
            try:
                while threads_alive():
                    Channel.begin(o, Channel.WRITING)
                    if threads_alive():
                        fun(self, Channel.data(o))
                    Channel.end(o, Channel.WRITING)
            except Exception as e:
                stop_threads()
                raise e
            finally:
                DEBUG("stopping source")

        new_thread(thread_loop)

    return new_fun


@simple_decorator
def sink(fun):
    "decorator for streaming data sinks in separate threads"

    def new_fun(self, i):
        try:
            i_shape = fun(self)
        except TypeError:
            i_shape = None
        DEBUG("declaring sink")
        Channel.reading(i, i_shape)

        def thread_loop():
            DEBUG("starting sink")
            try:
                while threads_alive():
                    Channel.begin(i, Channel.READING)
                    if threads_alive():
                        fun(self, Channel.data(i))
                    Channel.end(i, Channel.READING)
            except Exception as e:
                stop_threads()
                raise e
            finally:
                DEBUG("stopping sink")

        new_thread(thread_loop)

    return new_fun


def switched(decorator):
    """decorator transform for switched decorations.
    adds start_fun and stop_fun methods to class to control fun"""

    @simple_decorator
    def new_decorator(fun):
        event = new_event()

        def inner_fun(self, *args):
            if args:
                event.wait()
            if threads_alive():
                return fun(self, *args)

        def new_fun(self, *args):
            setattr(self, "start_%s" % fun.__name__, event.set)
            setattr(self, "stop_%s" % fun.__name__, event.clear)
            decorator(inner_fun)(self, *args)

        return new_fun

    return new_decorator


@simple_decorator
def finite_source(fun):
    """decorator for streaming finite data sources in separate threads.
    The source must return True if data was found and False otherwise."""
    started = new_event()
    stopped = new_event()

    def new_fun(self, o):
        setattr(self, "start_%s" % fun.__name__, started.set)
        setattr(self, "wait_%s" % fun.__name__, stopped.wait)
        try:
            o_shape = fun(self)
        except TypeError:
            o_shape = None
        DEBUG("declaring finite_source")
        Channel.writing(o, o_shape)

        def thread_loop():
            started.wait()
            DEBUG("starting finite_source")
            try:
                while threads_alive() and not stopped.isSet():
                    Channel.begin(o, Channel.WRITING)
                    if threads_alive():
                        if not fun(self, Channel.data(o)):
                            stopped.set()
                    Channel.end(o, Channel.WRITING)
            except Exception as e:
                stop_threads()
                raise e
            finally:
                DEBUG("stopping finite_source")

        new_thread(thread_loop)

    return new_fun


# ----( class transformations )------------------------------------------------


def Stream(Class):
    "streamifies two-way and one-way transform classes"

    class NewClass(Class):
        __doc__ = Class.__doc__

    NewClass.__name__ = Class.__name__
    for name in ["transform", "transform_fwd", "transform_bwd"]:
        if name in dir(Class):
            stream = transform(getattr(Class, name))
            name = name.replace("transform", "stream")
            stream.__name__ = name
            setattr(NewClass, name, stream)
    return NewClass


def Source(Class, name="read", naming="reading"):
    "streamifies source classes"

    class NewClass(Class):
        __doc__ = Class.__doc__

    NewClass.__name__ = Class.__name__
    stream = source(getattr(Class, name))
    stream.__name__ = naming
    setattr(NewClass, naming, stream)
    return NewClass


def FiniteSource(Class, name="read", naming="reading"):
    "streamifies finite source classes"

    class NewClass(Class):
        __doc__ = Class.__doc__

    NewClass.__name__ = Class.__name__
    stream = finite_source(getattr(Class, name))
    stream.__name__ = naming
    setattr(NewClass, naming, stream)
    return NewClass


def Sink(Class, name="write", naming="writing"):
    "streamifies sink classes"

    class NewClass(Class):
        __doc__ = Class.__doc__

    NewClass.__name__ = Class.__name__
    stream = sink(getattr(Class, name))
    stream.__name__ = naming
    setattr(NewClass, naming, stream)
    return NewClass


# ----( validation )-----------------------------------------------------------


def validate():
    try:
        Channel.validate()
    finally:
        stop_threads()


# ----( testing )--------------------------------------------------------------


def test_pipeline(length=5):
    import time

    class Input:
        @switched(source)
        def read(self, data_out):
            data_out[0] = input()
            print("|-- %s" % data_out[0])

        def start(self):
            self.start_read()

        def stop(self):
            self.stop_read()

    class Copy:
        @transform
        def transform(self, data_in, data_out):
            data_out[0] = data_in[0] + "@"
            print("%s --> %s" % (data_in[0], data_out[0]))

    class Print:
        @sink
        def write(self, data_out):
            print("%s --|" % data_out[0])

    def String():
        return Channel([""])

    print("building channels")
    c = [String() for _ in range(length + 1)]

    print("building nodes")
    i = Input()
    t = [Copy() for _ in range(length)]
    o = Print()

    print("connecting nodes with channels")
    i.read(c[0])
    for n in range(length):
        t[n].transform(c[n], c[n + 1])
    o.write(c[-1])
    validate()

    print("starting")
    i.start()
    time.sleep(2)
    print("stopping")
    i.stop()

    print("stopping threads")
    stop_threads()


def test_kill():
    def loop():
        while True:
            pass

    threading.Thread(target=loop).start()


if __name__ == "__main__":
    test_pipeline()
