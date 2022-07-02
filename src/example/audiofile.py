import kazoo as K


def test_play1(filename="test.wav", size=1024):
    file = K.transforms.AudioFile(filename, size)
    audio = K.transforms.Audio(size, reading=False)
    sound = K.transforms.Complexes(size)

    audio.start()
    while file.read(sound):
        audio.write(sound)
    audio.stop()


def test_play2(filename="test.mp3", size=1024):

    file = K.AudioFile(filename, size)
    audio = K.Audio(size, reading=False)
    sound = K.Complexes()

    file.reading(sound)
    audio.writing(sound)

    audio.start()
    file.start_read()
    file.wait_read()
    audio.stop()

    K.stop_threads()


def test_image1(
    infile="test.mp3", outfile="test.jpg", size_exponent=10, factor_exponent=1
):
    "batch sound --> image"

    s = K.transforms.Supergram(size_exponent, factor_exponent)

    small_size = s.small_size
    super_size = s.super_size
    length = s.super_size  # arbitrary

    print("reading sound file %s" % infile)
    sound = K.formats.read_mp3(infile, small_size * length, small_size)
    image = K.transforms.Reals(length, super_size)

    print("transforming data")
    for i in range(length):
        s.transform_fwd(sound[i, :], image[i, :])

    print("saving image %s" % outfile)
    image = K.util.energy_to_loudness(image + 1e-5)
    K.formats.write_image(image, outfile)


def test_image2(
    infile="test.mp3", outfile="test.jpg", size_exponent=10, factor_exponent=1
):
    "sequential sound --> image"

    s = K.transforms.Supergram(size_exponent, factor_exponent)
    a = K.transforms.AudioFile(infile, s.small_size)

    small_size = s.small_size
    super_size = s.super_size
    length = s.super_size  # arbitrary

    print("transforming data")
    sound = K.transforms.Complexes(small_size)
    image = K.transforms.Reals(length, super_size)
    for i in range(length):
        a.read(sound)
        s.transform_fwd(sound, image[i, :])

    print("saving image %s" % outfile)
    image = K.util.energy_to_loudness(image + 1e-5)
    K.formats.write_image(image, outfile)


def test_image3(infile="test.mp3", size_exponent=9, factor_exponent=1):
    "streaming sound --> image"

    s = K.Supergram(size_exponent, factor_exponent)

    outfile = infile[:-4]
    small_size = s.small_size
    super_size = s.super_size
    length = s.super_size  # arbitrary
    frame_rate = float(K.DEFAULT_SAMPLE_RATE) / small_size

    a = K.AudioFile(infile, small_size, small_size * length)
    l = K.Loudness(super_size, frame_rate)
    i = K.ImageBuffer(outfile, super_size, length)

    sound = K.Complexes()
    energy = K.Reals()
    loud = K.Reals()

    a.reading(sound)
    s.stream_fwd(sound, energy)
    l.stream_fwd(energy, loud)
    i.writing(loud)

    a.start_read()
    a.wait_read()

    import time

    time.sleep(0.5)  # HACK

    i.assemble()
    K.stop_threads()


def test_animate1(infiles=None, size_exponent=10, factor_exponent=2):
    "animation with Supergram"

    if infiles is None:
        infiles = ["test.mp3"]

    super = K.transforms.Supergram(size_exponent, factor_exponent)
    audio = K.transforms.Audio(super.small_size, reading=False)
    screen = K.transforms.Screen(1024 + 512, 1024)
    frame_rate = float(audio.rate) / super.small_size
    loud = K.transforms.Loudness(screen.height, frame_rate)

    view_portion = 0.5
    mapping = K.transforms.Reals(screen.height)
    for i in range(screen.height):
        mapping[i] = view_portion * (0.5 + i) / screen.height
    spline = K.transforms.Spline(screen.height, super.super_size, mapping)

    sound = K.transforms.Complexes(super.small_size)
    energy = K.transforms.Reals(super.super_size)
    scaled = K.transforms.Reals(spline.size_in)
    louds = K.transforms.Reals(spline.size_in)

    for infile in infiles:
        file = K.transforms.AudioFile(infile, super.small_size)
        audio.start()
        while file.read(sound):
            audio.write(sound)
            super.transform_fwd(sound, energy)
            spline.transform_bwd(energy, scaled)
            loud.transform_fwd(scaled, louds)
            screen.vertical_sweep(louds)
        audio.stop()


def test_animate2(infiles=None, small_exponent=9, large_exponent=11):
    "animation with Multigram"

    if infiles is None:
        infiles = ["test.mp3"]

    multi = K.transforms.Multigram(small_exponent, large_exponent)
    audio = K.transforms.Audio(multi.size_in, reading=False)
    screen = K.transforms.Screen(1024 + 512, 1024)
    frame_rate = float(audio.rate) / multi.size_in
    loud = K.transforms.Loudness(screen.height, frame_rate)

    view_portion = 0.5
    mapping = K.transforms.Reals(screen.height)
    for i in range(screen.height):
        mapping[i] = view_portion * (0.5 + i) / screen.height
    spline = K.transforms.Spline(screen.height, multi.size_out, mapping)

    sound = K.transforms.Complexes(multi.size_in)
    energy = K.transforms.Reals(multi.size_out)
    scaled = K.transforms.Reals(spline.size_in)
    louds = K.transforms.Reals(spline.size_in)

    for infile in infiles:
        file = K.transforms.AudioFile(infile, multi.size_in)
        audio.start()
        while file.read(sound):
            audio.write(sound)
            multi.transform_fwd(sound, energy)
            spline.transform_bwd(energy, scaled)
            loud.transform_fwd(scaled, louds)
            screen.vertical_sweep(louds)
        audio.stop()


def test_animate3(infiles=None, small_exponent=9, large_exponent=11):
    "pitch animation with Multigram"

    if infiles is None:
        infiles = ["test.mp3"]

    size_in = (1 << small_exponent) / 2
    size_out = 1 << 10
    frame_rate = float(K.DEFAULT_SAMPLE_RATE) / size_in

    audio = K.transforms.Audio(size_in, reading=False)
    multi = K.transforms.Multigram(small_exponent, large_exponent)
    scale = multi.pitch_scale(size_out)
    screen = K.transforms.Screen(size_out * 3 / 2, size_out)
    loud = K.transforms.Loudness(size_out, frame_rate)

    assert multi.size_in == size_in

    sound = K.transforms.Complexes(size_in)
    energy = K.transforms.Reals(multi.size_out)
    pitch = K.transforms.Reals(size_out)
    louds = K.transforms.Reals(size_out)

    for infile in infiles:
        file = K.transforms.AudioFile(infile, size_in)
        audio.start()
        while file.read(sound):
            audio.write(sound)
            multi.transform_fwd(sound, energy)
            scale.transform_fwd(energy, pitch)
            loud.transform_fwd(pitch, louds)
            screen.vertical_sweep(louds)
        audio.stop()


def test_animate4(infiles=None, small_exponent=9, large_exponent=11):
    "pitch animation with streaming Multigram"

    if infiles is None:
        infiles = ["test.mp3"]

    size_in = (1 << small_exponent) / 2
    size_out = 1 << 10
    frame_rate = float(K.DEFAULT_SAMPLE_RATE) / size_in

    file = K.AudioFile(infiles[0], size_in)
    split = K.Splitter()
    audio = K.Audio(size_in, reading=False)
    multi = K.Multigram(small_exponent, large_exponent)
    scale = multi.pitch_scale(size_out)
    screen = K.Screen(size_out * 3 / 2, size_out)
    loud = K.Loudness(size_out, frame_rate)

    assert multi.size_in == size_in

    sound1 = K.Complexes()
    sound2 = K.Complexes()
    sound3 = K.Complexes()
    energy = K.Reals()
    pitch = K.Reals()
    louds = K.Reals()

    file.reading(sound1)
    split.transform(sound1, (sound2, sound3))
    audio.writing(sound2)
    multi.stream_fwd(sound3, energy)
    scale.stream_fwd(energy, pitch)
    loud.stream_fwd(pitch, louds)
    screen.vertical_sweeping(louds)

    audio.start()
    file.start_read()
    file.wait_read()
    audio.stop()

    K.stop_threads()


def test_images1(infile="test.mp3", small_exponent=10, large_exponent=12, color=True):
    "buffered images with Multigram"

    outfile = infile[:-4]
    size_in = (1 << small_exponent) / 2
    frame_rate = float(K.DEFAULT_SAMPLE_RATE) / size_in

    file = K.transforms.AudioFile(infile, size_in)
    multi = K.transforms.Multigram(small_exponent, large_exponent)
    loud = K.transforms.Loudness(multi.size_out, frame_rate)
    image = K.transforms.ImageBuffer(outfile, multi.size_out, color=color)

    assert multi.size_in == size_in

    sound = K.transforms.Complexes(multi.size_in)
    energy = K.transforms.Reals(multi.size_out)
    louds = K.transforms.Reals(multi.size_out)

    while file.read(sound):
        multi.transform_fwd(sound, energy)
        loud.transform_fwd(energy, louds)
        image.write(louds)
    image.assemble()


def test_images2(
    infile="test.mp3", small_exponent=8, large_exponent=12, size_out=1 << 11, color=True
):
    "buffered pitch images with Multigram"

    outfile = infile[:-4]
    size_in = (1 << small_exponent) / 2
    frame_rate = float(K.DEFAULT_SAMPLE_RATE) / size_in

    file = K.transforms.AudioFile(infile, size_in)
    multi = K.transforms.Multigram(small_exponent, large_exponent)
    scale = multi.pitch_scale(size_out)
    loud = K.transforms.Loudness(size_out, frame_rate)
    image = K.transforms.ImageBuffer(outfile, size_out, color=color)

    assert multi.size_in == size_in

    sound = K.transforms.Complexes(multi.size_in)
    energy = K.transforms.Reals(multi.size_out)
    pitch = K.transforms.Reals(size_out)
    louds = K.transforms.Reals(size_out)

    while file.read(sound):
        multi.transform_fwd(sound, energy)
        scale.transform_fwd(energy, pitch)
        loud.transform_fwd(pitch, louds)
        image.write(louds)
    image.assemble()


def test_images3(
    infile="test.mp3", small_exponent=9, large_exponent=11, size_out=1 << 10, color=True
):
    "buffered pitch images with streaming Multigram"

    outfile = infile[:-4]

    multi = K.Multigram(small_exponent, large_exponent)
    file = K.AudioFile(infile, multi.size_in)
    scale = multi.pitch_scale(size_out)
    loud = K.Loudness(size_out, float(K.DEFAULT_SAMPLE_RATE) / multi.size_in)
    image = K.ImageBuffer(outfile, size_out, color=color)

    sound = K.Complexes()
    energy = K.Reals()
    pitch = K.Reals()
    louds = K.Reals()

    file.reading(sound)
    multi.stream_fwd(sound, energy)
    scale.stream_fwd(energy, pitch)
    loud.stream_fwd(pitch, louds)
    image.writing(louds)

    file.start_read()
    file.wait_read()

    import time

    time.sleep(0.5)  # HACK

    image.assemble()

    K.stop_threads()


def test_history1(
    infile="test.wav",
    small_exponent=9,
    large_exponent=10,
    size=1 << 9,
    length=1 << 9,
    density=1 << 7,
):
    "image of pitch history with Multigram"

    outfile = infile[:-4] + ".jpg"
    size_in = (1 << small_exponent) / 2
    frame_rate = float(K.DEFAULT_SAMPLE_RATE) / size_in

    file = K.transforms.AudioFile(infile, size_in)
    multi = K.transforms.Multigram(small_exponent, large_exponent)
    scale = multi.pitch_scale(size)
    loud = K.transforms.Loudness(size, frame_rate)
    hist = K.transforms.History(size, length, density)

    assert multi.size_in == size_in

    sound = K.transforms.Complexes(multi.size_in)
    energy = K.transforms.Reals(multi.size_out)
    pitch = K.transforms.Reals(size)
    louds = K.transforms.Reals(size)

    while file.read(sound) and not hist.full:
        multi.transform_fwd(sound, energy)
        scale.transform_fwd(energy, pitch)
        loud.transform_fwd(pitch, louds)
        hist.add(louds)
    if hist.full:
        print("history is full!")

    image = K.transforms.Reals(length, size)
    hist.get(image)
    K.formats.write_image(image, outfile)


def test_history2(
    infile="test.wav",
    small_exponent=9,
    large_exponent=12,
    size=1 << 10,
    length=1 << 9,
    density=1 << 7,
):
    "animation of pitch history with Multigram"

    size_in = (1 << small_exponent) / 2
    frame_rate = float(K.DEFAULT_SAMPLE_RATE) / size_in

    file = K.transforms.AudioFile(infile, size_in)
    multi = K.transforms.Multigram(small_exponent, large_exponent)
    scale = multi.pitch_scale(size)
    loud = K.transforms.Loudness(size, frame_rate)
    hist = K.transforms.History(size, length, density)
    screen = K.transforms.Screen(size, length)

    assert multi.size_in == size_in

    sound = K.transforms.Complexes(multi.size_in)
    energy = K.transforms.Reals(multi.size_out)
    pitch = K.transforms.Reals(size)
    louds = K.transforms.Reals(size)
    image = K.transforms.Reals(length, size)

    while file.read(sound):
        multi.transform_fwd(sound, energy)
        scale.transform_fwd(energy, pitch)
        loud.transform_fwd(pitch, louds)
        hist.transform(louds, image)
        screen.draw_vh(image)


def test_shepard1(
    infile="test.wav",
    small_exponent=9,
    large_exponent=11,
    size=1 << 10,
    zoom_factor=5,
    length=200,
):
    "animation of pitch history with Multigram"

    size_in = (1 << small_exponent) / 2
    frame_rate = float(K.DEFAULT_SAMPLE_RATE) / size_in

    file = K.transforms.AudioFile(infile, size_in)
    multi = K.transforms.Multigram(small_exponent, large_exponent)
    scale1 = multi.pitch_scale(size)
    loud = K.transforms.Loudness(size, frame_rate)
    scale2 = K.transforms.Shepard(size)
    zoom = K.transforms.Spline(12 * zoom_factor, 12)
    screen = K.transforms.Screen(length, 12 * zoom_factor)

    assert multi.size_in == size_in

    sound = K.transforms.Complexes(multi.size_in)
    energy = K.transforms.Reals(multi.size_out)
    pitch = K.transforms.Reals(size)
    louds = K.transforms.Reals(size)
    tone = K.transforms.Reals(12)
    zoomed = K.transforms.Reals(12 * zoom_factor)

    while file.read(sound):
        multi.transform_fwd(sound, energy)
        scale1.transform_fwd(energy, pitch)
        loud.transform_fwd(pitch, louds)
        scale2.transform_fwd(louds, tone)
        zoom.transform_bwd(tone, zoomed)
        zoomed /= zoomed.max()
        screen.vertical_sweep(zoomed)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    # test_play1()
    # test_image3()
    # test_animate4(args)
    # test_images2(args[0]) if args else test_images2()
    # K.network.debug_threads(test_images3)
    test_history2()
    # test_shepard1()
