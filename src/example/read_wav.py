import kazoo as K
from kazoo import formats


def read_spectrogram(exponent=10, wavname="test.wav"):
    """
    read a sound file, say a wav file;
    compute a spectrogram or reassigned spectrogram;
    write image to png file
    """

    width = 1 << exponent
    height = width
    size = height * width

    print("reading sound file")
    sound = formats.read_wav(wavname, size, width)
    image = K.Reals(height, width / 2)

    print("transforming data")
    s = K.Spectrogram(exponent)
    for i in range(height):
        s.transform_fwd(sound[i, :], image[i, :])

    print("saving image")
    image = formats.energy_to_loudness(image)
    formats.write_image(image, "test.png")


if __name__ == "__main__":
    read_spectrogram()
