
#include "audio.h"
#include <cstring>
#include <errno.h>

//----( PortAudio tools )-----------------------------------------------------

#define ASSERT_PA(err) {if ((err) != paNoError) {\
  ERROR( "PortAudio error: " << Pa_GetErrorText(err) );}}

#define ASSERT_STREAM(stream,err) {if ((err) != paNoError) {\
  Pa_AbortStream(stream);\
  Pa_CloseStream(stream);\
  ERROR( "PortAudio error: " << Pa_GetErrorText(err) );}}

#define ASSERT_IO(err) {if ((err) != paNoError) {\
  if (err & paInputOverflow) WARN("PortAudio Input Overflow"); \
  if (err & paOutputUnderflow) WARN("PortAudio Output Underflow"); \
  WARN( "PortAudio error: " << Pa_GetErrorText(err) ); }}

// the following allows interleaved PortAudio channels to be
// byte-equivalent to complex arrays, when num_channels = 2
#define PA_SAMPLE_TYPE  paFloat32

void print_device_info (PaDeviceIndex index)
{
  const PaDeviceInfo * device = Pa_GetDeviceInfo(index);
  PRINT(device->name);
  PRINT(device->maxInputChannels);
  PRINT(device->maxOutputChannels);
  PRINT(device->defaultLowInputLatency);
  PRINT(device->defaultLowOutputLatency);
  PRINT(device->defaultHighInputLatency);
  PRINT(device->defaultHighOutputLatency);
  PRINT(device->defaultSampleRate);
}

void print_host_api_info (PaDeviceIndex index)
{
  const PaHostApiInfo * api = Pa_GetHostApiInfo(index);
  PRINT(api->type);
  PRINT(api->name);
  PRINT(api->deviceCount);
  PRINT(api->defaultInputDevice);
  PRINT(api->defaultOutputDevice);
}

//----( audio device )--------------------------------------------------------

Audio::Audio (
    size_t frames_per_buffer,
    size_t sample_rate,
    bool reading,
    bool writing,
    bool stereo,
    bool verbose)

  : m_frames_per_buffer(frames_per_buffer),
    m_sample_rate(sample_rate),
    m_reading(reading),
    m_writing(writing),
    m_stereo(stereo),
    m_num_channels(stereo ? 2 : 1),
    m_active(false)
{
  ASSERT(reading or writing,
         "Audio must either read or write (or both)");

  if (verbose) LOG("PortAudio Initializing");
  ASSERT_PA(Pa_Initialize());

  const char * api_name = Pa_GetHostApiInfo(Pa_GetDefaultHostApi())->name;
  LOG("PortAudio using API " << api_name);

  if (verbose) {
    PaDeviceIndex num_devices = Pa_GetDeviceCount();
    LOG("PortAudio: " << num_devices << " devices available");
    for (PaDeviceIndex index = 0; index < num_devices; ++index) {
      LOG("PortAudio device #" << index << ":");
      print_device_info(index);
    }
  }

  if (m_reading) {
    m_input_parameters.device = Pa_GetDefaultInputDevice();
    ASSERT(m_input_parameters.device >= 0, "input device is busy");
    if (verbose) {
      LOG("PortAudio using input device #" << m_input_parameters.device);
    }

    m_input_parameters.channelCount = m_num_channels;
    m_input_parameters.sampleFormat = PA_SAMPLE_TYPE;
    m_input_parameters.suggestedLatency
      = Pa_GetDeviceInfo( m_input_parameters.device )->defaultLowInputLatency ;
    m_input_parameters.hostApiSpecificStreamInfo = NULL;
  }

  if (m_writing) {
    m_output_parameters.device = Pa_GetDefaultOutputDevice();
    ASSERT(m_output_parameters.device >= 0, "output device is busy");
    if (verbose) {
      LOG("PortAudio using output device #" << m_output_parameters.device);
    }

    m_output_parameters.channelCount = m_num_channels;
    m_output_parameters.sampleFormat = PA_SAMPLE_TYPE;
    m_output_parameters.suggestedLatency
      = Pa_GetDeviceInfo( m_output_parameters.device )->defaultLowOutputLatency;
    m_output_parameters.hostApiSpecificStreamInfo = NULL;
  }

  // setup stream
  ASSERT_STREAM(m_stream,
      Pa_OpenStream(
        &m_stream,
        m_reading ? &m_input_parameters : NULL,
        m_writing ? &m_output_parameters : NULL,
        m_sample_rate,
        m_frames_per_buffer,
        // paNoFlag,         //safe version
        paClipOff,        // unsafe version
        NULL,             // no callback, use blocking API
        NULL ));          // no callback, so no callback userData
}

Audio::~Audio ()
{
  ASSERT_STREAM(m_stream, Pa_CloseStream(m_stream));

  ASSERT_PA(Pa_Terminate());
  LOG("PortAudio Terminated");
}

void Audio::start ()
{
  ASSERT(not m_active, "tried to start Audio twice");
  ASSERT_STREAM(m_stream, Pa_StartStream(m_stream));
  m_active = true;
}

void Audio::read (Vector<complex> & samples)
{
  ASSERT(m_reading, "tried to read from write-only Audio");
  ASSERT_SIZE(samples, m_frames_per_buffer);

  if (m_active) {
    ASSERT_IO(Pa_ReadStream(m_stream, samples, m_frames_per_buffer));
  } else {
    WARN("tried to read from AudioStream while not active");
  }
  //DEBUG('>');
}

void Audio::write (const Vector<complex> & samples)
{
  ASSERT_SIZE(samples, m_frames_per_buffer);
  ASSERT(m_writing, "tried to write to read-only Audio");

  if (m_active) {
    ASSERT_IO(Pa_WriteStream(m_stream, samples, m_frames_per_buffer));
  } else {
    WARN("tried to write to AudioStream while not active");
  }
  //DEBUG('<');
}

void Audio::stop ()
{
  ASSERT(m_active, "tried to stop Audio twice");
  m_active = false;
  ASSERT_STREAM(m_stream, Pa_StopStream(m_stream));
}

//----( audio device thread )-------------------------------------------------

AudioThread::AudioThread (
    size_t frames_per_buffer,
    size_t sample_rate,
    bool reading,
    bool writing,
    bool stereo,
    bool verbose)

  : m_frames_per_buffer(frames_per_buffer),
    m_sample_rate(sample_rate),
    m_reading(reading),
    m_writing(writing),
    m_stereo(stereo),
    m_num_channels(stereo ? 2 : 1),
    m_active(false)
{
  ASSERT(reading or writing,
         "AudioThread must either read or write (or both)");

  LOG("PortAudio Initializing");
  ASSERT_PA(Pa_Initialize());

  PaDeviceIndex num_devices = Pa_GetDeviceCount();
  if (verbose) {
    LOG("----( audio devices )----");
    LOG("PortAudio: " << num_devices << " devices available");
    for (PaDeviceIndex index = 0; index < num_devices; ++index) {
        LOG("PortAudio device #" << index << ":");
        print_device_info(index);
    }
  }
  PaDeviceIndex input_device = Pa_GetDefaultInputDevice();
  PaDeviceIndex output_device = Pa_GetDefaultOutputDevice();

  PaHostApiIndex num_apis = Pa_GetHostApiCount();
  if (verbose) {
    LOG("----( host apis )----");
    LOG("PortAudio: " << num_apis << " apis available");
  }
  for (PaHostApiIndex index = 0; index < num_apis; ++index) {
    if (verbose) {
      LOG("PortAudio host api #" << index << ":");
      print_host_api_info(index);
    }

    // prefer jack if available
    const PaHostApiInfo * info = Pa_GetHostApiInfo(index);
    if (info->type == paJACK) {
      LOG("whoo hoo, using jack!");
      input_device = info->defaultInputDevice;
      output_device = info->defaultOutputDevice;
    }
  }

  if (verbose) LOG("----( streams )----");
  if (m_reading) {
    m_input_parameters.device = input_device;
    ASSERT(m_input_parameters.device >= 0, "input device is busy");
    if (verbose) {
      LOG("PortAudio using input device #" << m_input_parameters.device);
    }

    m_input_parameters.channelCount = m_num_channels;
    m_input_parameters.sampleFormat = PA_SAMPLE_TYPE;
    m_input_parameters.suggestedLatency
      = Pa_GetDeviceInfo( m_input_parameters.device )->defaultLowInputLatency ;
    m_input_parameters.hostApiSpecificStreamInfo = NULL;
  }

  if (m_writing) {
    m_output_parameters.device = output_device;
    ASSERT(m_output_parameters.device >= 0, "output device is busy");
    if (verbose) {
      LOG("PortAudio using output device #" << m_output_parameters.device);
    }

    m_output_parameters.channelCount = m_num_channels;
    m_output_parameters.sampleFormat = PA_SAMPLE_TYPE;
    m_output_parameters.suggestedLatency
      = Pa_GetDeviceInfo( m_output_parameters.device )->defaultLowOutputLatency;
    m_output_parameters.hostApiSpecificStreamInfo = NULL;
  }

  // setup stream
  ASSERT_STREAM(m_stream,
      Pa_OpenStream(
        &m_stream,
        m_reading ? &m_input_parameters : NULL,
        m_writing ? &m_output_parameters : NULL,
        m_sample_rate,
        m_frames_per_buffer,
        //paNoFlag,             //safe version
        paClipOff,            // unsafe version
        AudioThread_callback, // callback
        this));               // user data for callback
}

AudioThread::~AudioThread ()
{
  ASSERT_STREAM(m_stream, Pa_CloseStream(m_stream));

  ASSERT_PA(Pa_Terminate());
  LOG("PortAudio Terminated");
}

void AudioThread::start ()
{
  ASSERT(not m_active, "tried to start AudioThread twice");
  ASSERT_STREAM(m_stream, Pa_StartStream(m_stream));
  m_active = true;
}

void AudioThread::stop ()
{
  ASSERT(m_active, "tried to stop AudioThread twice");
  m_active = false;
  ASSERT_STREAM(m_stream, Pa_AbortStream(m_stream));
}

extern "C" int AudioThread_callback(
    const void * input,
    void * output,
    unsigned long frame_count,
    const PaStreamCallbackTimeInfo * time_info,
    PaStreamCallbackFlags status_flags,
    void * object)
{
  if (status_flags & paPrimingOutput) return 0; // wait for real input

  // TODO figure out how to use PortAudio's timestamps
  //Seconds time_in = Seconds(time_info.inputBufferAdcTime);
  //Seconds time_out = Seconds(time_info.outputBufferDacTime);

  AudioThread * thread = static_cast<AudioThread *>(object);
  if (thread->stereo()) {
    thread->process(
        static_cast<const complex * restrict>(input),
        static_cast<complex * restrict>(output),
        frame_count);
  } else {
    thread->process(
        static_cast<const float * restrict>(input),
        static_cast<float * restrict>(output),
        frame_count);
  }

  return thread->active() ? paContinue : paAbort;
}

//----( audio file )----------------------------------------------------------

AudioFile::AudioFile (string filename, bool stereo, size_t frame_size)
  : m_file(filename == "stdin" ? stdin : fopen(filename.c_str(), "rb")),
    m_stereo(stereo),
    m_num_channels(stereo ? 2 : 1),
    m_frame_size(frame_size),
    m_buffer(m_num_channels * m_frame_size),
    m_done(not m_file)
{
  ASSERT_LT(0, frame_size);

  if (m_file) {
    LOG("reading raw " << (stereo ? "stereo" : "mono") << " audio file "
        << filename);
  } else {
    WARN("could not read raw sound file " << filename);
  }
}

AudioFile::~AudioFile ()
{
  if (m_file and (m_file != stdin)) fclose(m_file);
}

void AudioFile::skip (size_t num_frames)
{
  int info = fseek(m_file, num_frames * sizeof(complex), SEEK_CUR);
  ASSERT(not info, "fseek failed with error " << strerror(errno));
}

void AudioFile::read_frame (Vector<float> & samples)
{
  ASSERT_SIZE(samples, m_frame_size);
  ASSERT(not stereo(), "tried to read mono samples from stereo AudioFile");

  read_frame(samples.data);
}

void AudioFile::read_frame (Vector<complex> & samples)
{
  ASSERT_SIZE(samples, m_frame_size);
  ASSERT(stereo(), "tried to read stereo samples from mono AudioFile");

  read_frame(Vector<float>(samples).data);
}

void AudioFile::read_frame (float * restrict samples)
{
  ASSERT(not done(), "tried to read from AudioFile after done()");

  const size_t num_samples = m_buffer.size;
  size_t num_to_read = num_samples;
  int16_t * restrict buffer = m_buffer;

  while (num_to_read) {

    if (feof(m_file)) {

      LOG("done reading raw sound file");
      m_done = true;

      if (m_file != stdin) fclose(m_file);
      m_file = NULL;

      bzero(buffer, sizeof(int16_t) * num_to_read);

      break;
    }

    size_t num_read = fread(buffer, sizeof(int16_t), num_to_read, m_file);
    num_to_read -= num_read;
    buffer += num_read;
  }

  const float scale = 1.0f / (1 << 15);
  buffer = m_buffer;
  for (size_t i = 0; i < num_samples; ++i) {
    samples[i] = scale * buffer[i];
  }
}

//----( simple functional api )-----------------------------------------------

template<class Sample>
inline bool read_audio_sample_ (
    const char * filename,
    Vector<Sample> & sound,
    size_t begin_frame)
{
  AudioFile file(filename);
  const size_t B = file.frame_size();
  Vector<Sample> buffer(B);

  sound.zero();

  if (begin_frame) file.skip(begin_frame);

  Sample * restrict destin = sound.begin();
  while (destin < sound.end()) {
    if (file.done()) return false;

    size_t b = sound.end() - destin;
    if (b >= B) {
      Vector<Sample> buffer(B, destin);
      file.read_frame(buffer);
      destin += B;
    }

    else {
      Vector<Sample> buffer(B);
      file.read_frame(buffer);
      memcpy(destin, buffer.data, b * sizeof(Sample));
      break;
    }
  }

  return true;
}

bool read_audio_sample (
    const char * filename,
    Vector<float> & sound,
    size_t begin_frame)
{
  return read_audio_sample_(filename, sound, begin_frame);
}

bool read_audio_sample (
    const char * filename,
    Vector<complex> & sound,
    size_t begin_frame)
{
  return read_audio_sample_(filename, sound, begin_frame);
}

