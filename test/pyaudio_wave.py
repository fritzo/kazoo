# from http://people.csail.mit.edu/hubert/pyaudio/#examples

import pyaudio
import wave


def test_wire():
    """A wire between input and output."""

    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = DEFAULT_SAMPLE_RATE
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=chunk,
    )

    print("* recording")
    for i in range(0, 44100 / chunk * RECORD_SECONDS):
        data = stream.read(chunk)
        stream.write(data, chunk)
    print("* done")

    stream.stop_stream()
    stream.close()
    p.terminate()


def test_play(filename="test.wav"):
    """Play a WAVE file."""

    chunk = 1024

    wf = wave.open(filename, "rb")

    p = pyaudio.PyAudio()

    # open stream
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )

    # read data
    data = wf.readframes(chunk)

    # play stream
    while data != "":
        stream.write(data)
        data = wf.readframes(chunk)

    stream.close()
    p.terminate()


if __name__ == "__main__":
    test_wire()
