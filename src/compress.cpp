
/** Video compression using FFV1 in libavcodec (R3)
 
  Comparison of codecs. see (R1) and (R2).
  * FFV1 is fast, efficient, and portable.
  * Lagarith is faster and better for video editing than FFV1
    but is Windows-only (libavcodec includes a decoder, but no encoder)
  * x264 implements losslessh.264 and is freely available, but is slower.
    However, Sandy Bridge chips have on-chip encoders (R4)

  (R1) Lossless Video Codecs Comparison 2007
    http://compression.ru/video/codec_comparison/lossless_codecs_2007_en.html
  (R2) http://en.wikipedia.org/wiki/Libavcodec
  (R3) Description of the FFV1 Video Codec
    http://www.ffmpeg.org/~michael/ffv1.html
  (R4) http://en.wikipedia.org/wiki/H264

  (N1) FFV1 supports the following pixel formats:
    (from http://wiki.multimedia.cx/index.php?title=FFV1)
    YUV420P
    YUV444P
    YUV422P
    YUV411P
    YUV410P
    RGB32
    YUV420P16
    YUV422P16
    YUV444P16
*/

#include "compress.h"
#include "threads.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>

// HACK to avoid compile error for undefined macro UINT64_C
// see http://code.google.com/p/ffmpegsource/source/detail?r=311
#define __STDC_CONSTANT_MACROS

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/mathematics.h>

int lock_manager_callback (void **ppMutex, enum AVLockOp op)
{
  Mutex * & pMutex = (Mutex*&)(*ppMutex);

  switch(op) {
    case AV_LOCK_CREATE: pMutex = new Mutex; break;
    case AV_LOCK_OBTAIN: pMutex->lock(); break;
    case AV_LOCK_RELEASE: pMutex->unlock(); break;
    case AV_LOCK_DESTROY: delete pMutex; break;
  }

  return 0;
}

} // extern "C"

// if this is too low, encoder will complain about "buffer underflow"
//#define DEFAULT_BIT_RATE (320 * 240 * DEFAULT_VIDEO_FRAMERATE * 1)
#define DEFAULT_BIT_RATE (0)

namespace Video
{

using namespace Streaming;

const CodecID codec_id = CODEC_ID_FFV1;
//const CodecID codec_id = CODEC_ID_MPEG1VIDEO;
//const CodecID codec_id = CODEC_ID_MPEG4;
//const CodecID codec_id = CODEC_ID_H264;

static void initialize_libavcodec ()
{
  static Mutex mutex;
  mutex.lock();

  static bool initialized = false;
  if (not initialized) {
    initialized = true;
    av_register_all(); // must be called before using libavcodec
    av_lockmgr_register(lock_manager_callback);
  }

  mutex.unlock();
}

static AVFrame * alloc_picture (PixelFormat pix_fmt, int width, int height)
{
  AVFrame * picture;
  uint8_t * picture_buf;
  int size;

  picture = avcodec_alloc_frame();
  if (!picture) return NULL;

  size = avpicture_get_size(pix_fmt, width, height);
  picture_buf = (uint8_t *)av_malloc(size);
  if (!picture_buf) {
    av_free(picture);
    return NULL;
  }

  avpicture_fill((AVPicture *) picture, picture_buf, pix_fmt, width, height);

  return picture;
}

//----( encoding formatted streams )------------------------------------------

// this is adapted from
// http://cekirdek.pardus.org.tr/~ismail/ffmpeg-docs/output-example_8c-source.html

class EncoderGuts
{
  AVOutputFormat * m_format;
  AVFormatContext * m_format_context;

  AVStream * m_video_stream;
  AVCodecContext * m_codec_context;
  AVCodec * m_codec;

  AVFrame * m_picture;
  uint8_t * m_video_outbuf;
  size_t m_video_outbuf_size;
  size_t m_frame_count;

public:

  const Rectangle shape;
  const size_t framerate;

  EncoderGuts (
      string filename,
      Rectangle shape,
      size_t framerate = DEFAULT_VIDEO_FRAMERATE);
  ~EncoderGuts ();

  void add_video_stream (CodecID codec_id);
  void open_video ();
  void close_video ();
  void push (const Yuv420p8Image & image);
  void push (const Mono8Image & image);
  void encode_image ();
};

EncoderGuts::EncoderGuts (string filename, Rectangle wh, size_t fps)
  : m_frame_count(0),
    shape(wh),
    framerate(fps)
{
  LOG("compressing to " << filename);

  initialize_libavcodec();

  m_format = av_guess_format("avi", NULL, NULL);
  ASSERT(m_format, "Could not find suitable output m_format");

  // set audio & video codecs
  m_format->audio_codec = CODEC_ID_NONE;
  m_format->video_codec = codec_id;

  // allocate the output media context
  m_format_context = avformat_alloc_context();
  ASSERT(m_format_context, "failed to allocate m_format context");
  m_format_context->oformat = m_format;
  snprintf(m_format_context->filename,
      sizeof(m_format_context->filename),
      "%s",
      filename.c_str());

  // add the video stream using the format's codec and initialize the codec
  ASSERT(m_format->video_codec != CODEC_ID_NONE, "no video stream");
  add_video_stream(m_format->video_codec);

  // set the output parameters (must be done even if no parameters).
  ASSERT(av_set_parameters(m_format_context, NULL) >= 0,
      "Invalid output m_format parameters");

  dump_format(m_format_context, 0, filename.c_str(), 1);

  // now that all the parameters are set, we can open the
  // video codec and allocate the necessary encode buffers
  open_video();

  // open the output file, if needed
  if (!(m_format->flags & AVFMT_NOFILE)) {
    ASSERT(url_fopen(&m_format_context->pb, filename.c_str(), URL_WRONLY) >= 0,
        "Could not open url " << filename);
  }

  // write the stream header, if any
  av_write_header(m_format_context);
}

// add a video output stream
void EncoderGuts::add_video_stream (CodecID codec_id)
{
  m_video_stream = av_new_stream(m_format_context, 0);
  ASSERT(m_video_stream, "Could not alloc stream");

  m_codec_context = m_video_stream->codec;
  m_codec_context->codec_id = codec_id;
  m_codec_context->codec_type = AVMEDIA_TYPE_VIDEO;

  // put sample parameters
  m_codec_context->bit_rate = DEFAULT_BIT_RATE;

  // resolution must be a multiple of two
  m_codec_context->width = shape.height();
  m_codec_context->height = shape.width();

  // time base: this is the fundamental unit of time (in seconds) in terms
  // of which frame timestamps are represented. for fixed-fps content,
  // timebase should be 1/framerate and timestamp increments should be
  // identically 1.
  m_codec_context->time_base.den = framerate;
  m_codec_context->time_base.num = 1;
  m_codec_context->pix_fmt = PIX_FMT_YUV420P;

  // some formats want stream headers to be separate
  if (m_format_context->oformat->flags & AVFMT_GLOBALHEADER) {
    m_codec_context->flags |= CODEC_FLAG_GLOBAL_HEADER;
  }
}

void EncoderGuts::open_video ()
{
  m_codec_context = m_video_stream->codec;

  // find the video encoder
  m_codec = avcodec_find_encoder(m_codec_context->codec_id);
  ASSERT(m_codec, "encoding m_codec not found");

  // open the m_codec
  ASSERT(avcodec_open(m_codec_context, m_codec) >= 0, "could not open m_codec");

  ASSERT(!(m_format_context->oformat->flags & AVFMT_RAWPICTURE),
      "raw video encoding is not supported");

  // allocate output buffer
  // XXX: API change will be done

  // buffers passed into lav* can be allocated any way you prefer,
  // as long as they're aligned enough for the architecture, and
  // they're freed appropriately (such as using av_free for buffers
  // allocated with av_malloc)
  m_video_outbuf_size = 200000;
  m_video_outbuf = (uint8_t *)av_malloc(m_video_outbuf_size);

  // allocate the encoded raw picture
  m_picture = alloc_picture(
      m_codec_context->pix_fmt,
      m_codec_context->width,
      m_codec_context->height);
  ASSERT(m_picture, "Could not allocate picture");

  ASSERT(m_codec_context->pix_fmt == PIX_FMT_YUV420P,
      "only format YUV420P is supported");
}

void EncoderGuts::push (const Yuv420p8Image & image)
{
  ASSERT_SIZE(image.y, shape.size());

  // TODO avoid a copy by creating an alias AVFrame and filling,
  // as in alloc_picture
  memcpy(m_picture->data[0], image.data, image.size);

  encode_image();
}

void EncoderGuts::push (const Mono8Image & image)
{
  ASSERT_SIZE(image, shape.size());

  // TODO avoid a copy by creating an alias AVFrame and filling,
  // as in alloc_picture
  memcpy(m_picture->data[0], image.data, image.size);
  memset(m_picture->data[1], 127, image.size / 4);
  memset(m_picture->data[2], 127, image.size / 4);

  encode_image();
}

void EncoderGuts::encode_image ()
{
  // encode the image
  int out_size = avcodec_encode_video(
      m_codec_context,
      m_video_outbuf,
      m_video_outbuf_size,
      m_picture);

  // if zero size, it means the image was buffered
  if (out_size > 0) {
    AVPacket pkt;
    av_init_packet(&pkt);

    if (m_codec_context->coded_frame->pts != (int64_t)AV_NOPTS_VALUE) {
      pkt.pts = av_rescale_q(
          m_codec_context->coded_frame->pts,
          m_codec_context->time_base,
          m_video_stream->time_base);
    }

    if (m_codec_context->coded_frame->key_frame) {
      pkt.flags |= AV_PKT_FLAG_KEY;
    }

    pkt.stream_index = m_video_stream->index;
    pkt.data = m_video_outbuf;
    pkt.size = out_size;

    // write the compressed frame in the media file
    int info = av_interleaved_write_frame(m_format_context, &pkt);
    ASSERT(info == 0, "Error while writing video frame");
  }

  ++m_frame_count;
}

void EncoderGuts::close_video ()
{
  avcodec_close(m_video_stream->codec);
  av_free(m_picture->data[0]);
  av_free(m_picture);
  av_free(m_video_outbuf);
}

EncoderGuts::~EncoderGuts ()
{
  LOG("encoded " << m_frame_count << " frames");

  // write the trailer, if any.  the trailer must be written
  // before you close the CodecContexts open when you wrote the
  // header; otherwise write_trailer may try to use memory that
  // was freed on av_codec_close()
  av_write_trailer(m_format_context);

  close_video();

  // free the streams (only one in our example, but generally multiple)
  ASSERT_EQ(m_format_context->nb_streams, 1);
  av_freep(&m_format_context->streams[0]->codec);
  av_freep(&m_format_context->streams[0]);

  if (!(m_format->flags & AVFMT_NOFILE)) {
    // close the output file
    url_fclose(m_format_context->pb);
  }

  // free the stream
  av_free(m_format_context);
}

} // namespace Video

//----------------------------------------------------------------------------

namespace Streaming
{

//----( encoder )-------------------------------------------------------------

VideoEncoder::VideoEncoder (string filename, Rectangle shape, size_t framerate)
  : m_guts(new Video::EncoderGuts(filename, shape, framerate))
{
}

VideoEncoder::~VideoEncoder ()
{
  delete m_guts;
}

void VideoEncoder::push (Seconds time, const Yuv420p8Image & image)
{
  m_guts->push(image);
}

void VideoEncoder::push (Seconds time, const Mono8Image & image)
{
  m_guts->push(image);
}

void VideoEncoder::push (const Yuv420p8Image & image) { m_guts->push(image); }
void VideoEncoder::push (const Mono8Image & image) { m_guts->push(image); }

//----( decoding to ram )-----------------------------------------------------

// this is adapted from http://dranger.com/ffmpeg/tutorial01.html

VideoFile::VideoFile (string filename)
  : m_shape(0,0),
    m_framerate(0)
{
  Video::initialize_libavcodec();

  LOG("decompressing from " << filename << "...");

  AVFormatContext * format_context;
  ASSERT(not av_open_input_file(
        & format_context,
        filename.c_str(),
        NULL, 0, NULL),
      "failed to open input file " << filename);

  ASSERT(av_find_stream_info(format_context) >= 0,
      "failed to find stream information");

  // we could search through, but instead assume there is only one stream
  ASSERT_EQ(format_context->nb_streams, 1);
  int video_stream_index = 0;

  ASSERT_EQ(
      format_context->streams[video_stream_index]->codec->codec_type,
      CODEC_TYPE_VIDEO);

  AVCodecContext * codec_context
    = format_context->streams[video_stream_index]->codec;
  m_shape = Rectangle(codec_context->height, codec_context->width);
  m_framerate = 1.0f * codec_context->time_base.den
                     / codec_context->time_base.num;
  LOG(" video format: " << m_shape.height()
      << " x " << m_shape.width()
      << " @ " << m_framerate << "Hz");

  const CodecID codec_id = codec_context->codec_id;
  AVCodec * codec = avcodec_find_decoder(codec_id);
  ASSERT(codec, "decoding codec not found");

  ASSERT(avcodec_open(codec_context, codec) >= 0, "could not open codec");

  float dt = 1.0f / m_framerate;
  Seconds time = Seconds::now();

  AVFrame * picture = avcodec_alloc_frame();
  AVPacket packet;
  while (av_read_frame(format_context, & packet) >= 0) {

    if (packet.stream_index == video_stream_index) {

      int frame_finished;
      avcodec_decode_video2(codec_context, picture, & frame_finished, & packet);

      if (frame_finished) {

        Yuv420p8Image * image = new Yuv420p8Image(m_shape.size());

        size_t I = m_shape.width();
        size_t J = m_shape.height();
        size_t L0 = picture->linesize[0];
        if (L0 == J) {
          memcpy(image->data, picture->data[0], image->size);
        } else {
          for (size_t i = 0; i < I; ++i) {
            memcpy(image->data + J * i, picture->data[0] + L0 * i, J);
          }
          size_t L1 = picture->linesize[1];
          size_t L2 = picture->linesize[2];
          for (size_t i = 0; i < I/2; ++i) {
            memcpy(image->data + J * I + J/2 * i,
                picture->data[1] + L1 * i,
                J/2);
            memcpy(image->data + J * I * 5/4 + J/2 * i,
                picture->data[2] + L2 * i,
                J/2);
          }
        }

        m_frames.push_back(Frame(time, image));
        time += dt;
      }
    }

    av_free_packet(& packet);
  }

  av_free(picture);

  avcodec_close(codec_context);

  av_close_input_file(format_context);

  double size_bytes = m_frames.size() * 1.5 * m_shape.size();
  double size_mb = size_bytes / (1<<20);
  LOG("...decoded " << m_frames.size() << " frames totaling "
      << roundu(size_mb) << "MB");
}

void VideoFile::clear ()
{
  for (iterator i = begin(); i != end(); ++i) {
    delete i->image;
  }

  m_frames.clear();
}

void VideoFile::dump_to (VideoSequence & seq)
{
  for (iterator i = begin(); i != end(); ++i) {
    seq.add(i->image);
  }

  m_frames.clear(); // do not delete
}

//----( video sequence )------------------------------------------------------

VideoSequence::VideoSequence (Rectangle shape)
  : m_shape(shape),
    m_file(0),
    m_time(0)
{
}

VideoSequence::VideoSequence (string filename)
  : m_shape(0,0),
    m_file(0),
    m_time(0)
{
  add_file(filename);
}

const Yuv420p8Image * VideoSequence::get (size_t i) const
{
  if (size() <= i) return NULL;
  return m_frames[i].image;
}

const Yuv420p8Image * VideoSequence::maybe_prev (size_t i) const
{
  if ((i == 0) or (size() <= i)) return NULL;
  if (m_frames[i-1].file != m_frames[i].file) return NULL;
  if (m_frames[i-1].time != m_frames[i].time - 1) return NULL;
  return m_frames[i-1].image;
}

const Yuv420p8Image * VideoSequence::maybe_next (size_t i) const
{
  if (size() <= i+1) return NULL;
  if (m_frames[i+1].file != m_frames[i].file) return NULL;
  if (m_frames[i+1].time != m_frames[i].time + 1) return NULL;
  return m_frames[i+1].image;
}

void VideoSequence::add (const Yuv420p8Image * image)
{
  ASSERT_SIZE(image->y, m_shape.size());

  m_frames.push_back(Frame(m_file, m_time++, image));
}

void VideoSequence::add_file (VideoFile & file)
{
  if (m_frames.empty()) {
    m_shape = file.shape();
  } else {
    ASSERT_EQ(m_shape, file.shape());
  }

  m_frames.reserve(m_frames.size() + file.frames().size());

  file.dump_to(*this);
  ++m_file;
  m_time = 0;
}

void VideoSequence::add_files (std::vector<string> filenames)
{
  const size_t num_files = filenames.size();
  Mutex mutex;

  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < num_files; ++i) {

    VideoFile file(filenames[i]); // all the work is done here

    mutex.lock();
    add_file(file);
    mutex.unlock();
  }
}

void VideoSequence::sort ()
{
  bool already_sorted = true;

  for (size_t i = 1, I = size(); (i < I) and already_sorted; ++i) {

    already_sorted = m_frames[i-1] < m_frames[i];
  }

  if (not already_sorted) {
    std::sort(m_frames.begin(), m_frames.end());
  }
}

void VideoSequence::shuffle ()
{
  std::random_shuffle(m_frames.begin(), m_frames.end());
}

void VideoSequence::clear ()
{
  typedef Frames::iterator Auto;
  for (Auto i = m_frames.begin(); i != m_frames.end(); ++i) {
    delete i->image;
  }

  m_frames.clear();
  m_file = 0;
  m_time = 0;
}

//----( video player )--------------------------------------------------------

VideoPlayer::VideoPlayer (const VideoFile & file, float speed)
  : TimedThread(file.framerate() * speed),
    m_file(file),
    m_pos(m_file.begin()),
    m_time(Seconds::now()),
    m_timestep(1 / file.framerate()),
    out("VideoPlayer.out", file.shape()),
    mono_out("VideoPlayer.mono_out", file.shape())
{}

void VideoPlayer::step ()
{
  if (m_pos == m_file.end()) return;

  const Yuv420p8Image & image = *m_pos->image;

  if (out) out.push(m_time, image);
  if (mono_out) mono_out.push(m_time, image.y);

  ++m_pos;
  m_time += m_timestep;
}

} // namespace Streaming

