
/** compress_test.cpp

  The initial codec_context version of this test
  is adapted from ffmpeg's example
    /usr/share/doc/libavcodec-dev/examples/api-example.codec_context.gz
*/

#include "common.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <SDL.h>
#include <SDL_thread.h>

#ifdef __MINGW32__
#undef main /* Prevents SDL from overriding main() */
#endif

// HACK to avoid compile error for undefined macro UINT64_C
// see http://code.google.com/p/ffmpegsource/source/detail?r=311
#define __STDC_CONSTANT_MACROS

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/mathematics.h>
}

#define INBUF_SIZE 4096

// 2 seconds stream duration
#define STREAM_DURATION   2.0
#define STREAM_FRAME_RATE 125 // 125 images/s
#define STREAM_NB_FRAMES  ((int)(STREAM_DURATION * STREAM_FRAME_RATE))

// if this is too low, encoder will complain about "buffer underflow"
//#define DEFAULT_BIT_RATE (320 * 240 * STREAM_FRAME_RATE * 1)
#define DEFAULT_BIT_RATE (0)

const CodecID codec_id = CODEC_ID_FFV1;
//const CodecID codec_id = CODEC_ID_MPEG1VIDEO;
//const CodecID codec_id = CODEC_ID_MPEG4;
//const CodecID codec_id = CODEC_ID_H264;

//----( encoding formatted streams )------------------------------------------

// this is adapted from
// http://cekirdek.pardus.org.tr/~ismail/ffmpeg-docs/output-example_8c-source.html

AVFrame * picture;
uint8_t * video_outbuf;
int frame_count, video_outbuf_size;

// add a video output stream
static AVStream * add_video_stream (
    AVFormatContext * format_context,
    enum CodecID codec_id)
{
  AVCodecContext *codec_context;
  AVStream *st;

  st = av_new_stream(format_context, 0);
  ASSERT(st, "Could not alloc stream");

  codec_context = st->codec;
  codec_context->codec_id = codec_id;
  codec_context->codec_type = AVMEDIA_TYPE_VIDEO;

  // put sample parameters
  codec_context->bit_rate = DEFAULT_BIT_RATE;
  PRINT(codec_context->bit_rate);

  // resolution must be a multiple of two
  codec_context->width = 320;
  codec_context->height = 240;

  // time base: this is the fundamental unit of time (in seconds) in terms
  // of which frame timestamps are represented. for fixed-fps content,
  // timebase should be 1/framerate and timestamp increments should be
  // identically 1.
  codec_context->time_base.den = STREAM_FRAME_RATE;
  codec_context->time_base.num = 1;
  codec_context->gop_size = 12; // emit one intra frame every 12 frames at most
  codec_context->pix_fmt = PIX_FMT_YUV420P;

  if (codec_context->codec_id == CODEC_ID_MPEG2VIDEO) {
    // just for testing, we also add B frames
    codec_context->max_b_frames = 2;
  }

  if (codec_context->codec_id == CODEC_ID_MPEG1VIDEO){
    // Needed to avoid using macroblocks in which some coeffs overflow.
    // This does not happen with normal video, it just happens here as
    // the motion of the chroma plane does not match the luma plane.
    codec_context->mb_decision=2;
  }

  // some formats want stream headers to be separate
  if (format_context->oformat->flags & AVFMT_GLOBALHEADER) {
    codec_context->flags |= CODEC_FLAG_GLOBAL_HEADER;
  }

  return st;
}

static AVFrame *alloc_picture(enum PixelFormat pix_fmt, int width, int height)
{
  AVFrame *picture;
  uint8_t *picture_buf;
  int size;

  picture = avcodec_alloc_frame();
  if (!picture) return NULL;

  size = avpicture_get_size(pix_fmt, width, height);
  picture_buf = (uint8_t *)av_malloc(size);
  if (!picture_buf) {
    av_free(picture);
    return NULL;
  }

  avpicture_fill(
      (AVPicture *)picture,
      picture_buf,
      pix_fmt,
      width,
      height);

  return picture;
}

static void open_video (AVFormatContext *format_context, AVStream *st)
{
  AVCodec *codec;
  AVCodecContext *codec_context;

  codec_context = st->codec;

  // find the video encoder
  codec = avcodec_find_encoder(codec_context->codec_id);
  ASSERT(codec, "encoding codec not found");

  // open the codec
  ASSERT(avcodec_open(codec_context, codec) >= 0, "could not open codec");

  video_outbuf = NULL;
  ASSERT(!(format_context->oformat->flags & AVFMT_RAWPICTURE),
      "raw video encoding is not supported");

  // allocate output buffer
  // XXX: API change will be done

  // buffers passed into lav* can be allocated any way you prefer,
  // as long as they're aligned enough for the architecture, and
  // they're freed appropriately (such as using av_free for buffers
  // allocated with av_malloc)
  video_outbuf_size = 200000;
  video_outbuf = (uint8_t *)av_malloc(video_outbuf_size);

  // allocate the encoded raw picture
  picture = alloc_picture(
      codec_context->pix_fmt,
      codec_context->width,
      codec_context->height);
  ASSERT(picture, "Could not allocate picture");

  ASSERT(codec_context->pix_fmt == PIX_FMT_YUV420P,
      "only format YUV420P is supported");
}

// prepare a dummy image
static void fill_yuv_image(
    AVFrame *pict,
    int frame_index,
    int width,
    int height)
{
  int x, y, i;

  i = frame_index;

  // Y
  for(y=0;y<height;y++) {
    for(x=0;x<width;x++) {
      pict->data[0][y * pict->linesize[0] + x] = x + y + i * 3;
    }
  }

  // Cb and Cr
  for(y=0;y<height/2;y++) {
    for(x=0;x<width/2;x++) {
      pict->data[1][y * pict->linesize[1] + x] = 128 + y + i * 2;
      pict->data[2][y * pict->linesize[2] + x] = 64 + x + i * 5;
    }
  }
}

static void write_video_frame (AVFormatContext * format_context, AVStream * st)
{
  int out_size, ret;
  AVCodecContext *codec_context;

  codec_context = st->codec;

  if (frame_count >= STREAM_NB_FRAMES) {
    // no more frame to compress. The codec has a latency of a few
    // frames if using B frames, so we get the last frames by
    // passing the same picture again

  } else {

    ASSERT(codec_context->pix_fmt == PIX_FMT_YUV420P,
        "only the YUV420P format is supported");

    fill_yuv_image(
        picture,
        frame_count,
        codec_context->width,
        codec_context->height);
  }

  ASSERT(!(format_context->oformat->flags & AVFMT_RAWPICTURE),
      "raw video encoding is not supported");

  // encode the image
  out_size = avcodec_encode_video(
      codec_context,
      video_outbuf,
      video_outbuf_size,
      picture);

  // if zero size, it means the image was buffered
  if (out_size > 0) {
    AVPacket pkt;
    av_init_packet(&pkt);

    if (int(codec_context->coded_frame->pts) != int(AV_NOPTS_VALUE)) {
      pkt.pts = av_rescale_q(
          codec_context->coded_frame->pts,
          codec_context->time_base,
          st->time_base);
    }

    if (codec_context->coded_frame->key_frame) {
      pkt.flags |= AV_PKT_FLAG_KEY;
    }

    pkt.stream_index= st->index;
    pkt.data= video_outbuf;
    pkt.size= out_size;

    // write the compressed frame in the media file
    ret = av_interleaved_write_frame(format_context, &pkt);

  } else {

    ret = 0;
  }

  ASSERT(ret == 0, "Error while writing video frame");

  frame_count++;
}

static void close_video (AVFormatContext * format_context, AVStream * st)
{
  avcodec_close(st->codec);
  av_free(picture->data[0]);
  av_free(picture);
  av_free(video_outbuf);
}

void test_encode_stream (const char *filename)
{
  AVOutputFormat *format;
  AVFormatContext *format_context;
  AVStream *video_st;
  double video_pts;

  // auto detect the output format from the name. default is mpeg. 
  format = av_guess_format(NULL, filename, NULL);
  if (!format) {
    LOG("Could not deduce output format from file extension: using MPEG.");
    format = av_guess_format("mpeg", NULL, NULL);
  }
  if (!format) {
    fprintf(stderr, "Could not find suitable output format\n");
    exit(1);
  }

  // set audio & video codecs
  format->audio_codec = CODEC_ID_NONE;
  format->video_codec = codec_id;

  // allocate the output media context
  format_context = avformat_alloc_context();
  ASSERT(format_context, "failed to allocate format context");
  format_context->oformat = format;
  snprintf(format_context->filename,
      sizeof(format_context->filename),
      "%s",
      filename);

  // add the video stream using the format's codec and initialize the codec
  ASSERT(format->video_codec != CODEC_ID_NONE, "no video stream");
  video_st = add_video_stream(format_context, format->video_codec);

  // set the output parameters (must be done even if no parameters).
  ASSERT(av_set_parameters(format_context, NULL) >= 0,
      "Invalid output format parameters");

  dump_format(format_context, 0, filename, 1);

  // now that all the parameters are set, we can open the
  // video codec and allocate the necessary encode buffers
  open_video(format_context, video_st);

  // open the output file, if needed
  if (!(format->flags & AVFMT_NOFILE)) {
    ASSERT(url_fopen(&format_context->pb, filename, URL_WRONLY) >= 0,
        "Could not open url " <<  filename);
  }

  // write the stream header, if any
  av_write_header(format_context);

  for (;;) {

    video_pts = (double)video_st->pts.val
              * video_st->time_base.num
              / video_st->time_base.den;

    if (video_pts >= STREAM_DURATION) break;

    write_video_frame(format_context, video_st);
  }

  // write the trailer, if any.  the trailer must be written
  // before you close the CodecContexts open when you wrote the
  // header; otherwise write_trailer may try to use memory that
  // was freed on av_codec_close()
  av_write_trailer(format_context);

  close_video(format_context, video_st);

  // free the streams (only one in our example, but generally multiple)
  ASSERT_EQ(format_context->nb_streams, 1);
  av_freep(&format_context->streams[0]->codec);
  av_freep(&format_context->streams[0]);

  if (!(format->flags & AVFMT_NOFILE)) {
    // close the output file
    url_fclose(format_context->pb);
  }

  // free the stream
  av_free(format_context);
}

//----( decoding formatted streams )------------------------------------------

// this is adapted from http://dranger.com/ffmpeg/tutorial01.html
void test_decode_stream (const char * filename)
{
  AVFormatContext * format_context;
  ASSERT(not av_open_input_file(&format_context, filename, NULL, 0, NULL),
      "failed to open input file " << filename);

  ASSERT(av_find_stream_info(format_context) >= 0,
      "failed to find stream information");

  // we could search through, but instead assume there is only one stream
  ASSERT_EQ(format_context->nb_streams, 1);
  int video_stream = 0;

  ASSERT_EQ(
      format_context->streams[video_stream]->codec->codec_type,
      AVMEDIA_TYPE_VIDEO);

  AVCodecContext * codec_context
    = format_context->streams[video_stream]->codec;
  //ASSERT_EQ(codec_context->width, 320);
  //ASSERT_EQ(codec_context->height, 240);

  const CodecID codec_id = codec_context->codec_id;
  AVCodec * codec = avcodec_find_decoder(codec_id);
  ASSERT(codec, "decoding codec not found");

  ASSERT(avcodec_open(codec_context, codec) >= 0, "could not open codec");

  AVFrame * picture = avcodec_alloc_frame();
  AVFrame * picture_rgb = avcodec_alloc_frame();

  int frame_finished;
  AVPacket packet;


  SDL_Overlay     *bmp;
  SDL_Surface     *screen;
  SDL_Rect        rect;
  SDL_Event       event;

  ASSERT(!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER),
    "Could not initialize SDL - " << SDL_GetError());

  // Make a screen to put our video
#ifndef __DARWIN__
  screen = SDL_SetVideoMode(codec_context->width, codec_context->height, 0, 0);
#else
  screen = SDL_SetVideoMode(codec_context->width, codec_context->height, 24, 0);
#endif
  ASSERT(screen, "SDL: could not set video mode - exiting");

  // Allocate a place to put our YUV image on that screen
  bmp = SDL_CreateYUVOverlay(
      codec_context->width,
      codec_context->height,
      SDL_YV12_OVERLAY,
      screen);

  static struct SwsContext *img_convert_ctx;
  PixelFormat dst_pix_fmt = PIX_FMT_YUV420P;

  while (av_read_frame(format_context, &packet)>=0) {

    if (packet.stream_index == video_stream) {

      avcodec_decode_video2(codec_context, picture, &frame_finished, &packet);

      if (frame_finished) {

        SDL_LockYUVOverlay(bmp);

        AVPicture pict;
        pict.data[0] = bmp->pixels[0];
        pict.data[1] = bmp->pixels[2];
        pict.data[2] = bmp->pixels[1];

        pict.linesize[0] = bmp->pitches[0];
        pict.linesize[1] = bmp->pitches[2];
        pict.linesize[2] = bmp->pitches[1];

        // Convert the image into YUV format that SDL uses
        if (img_convert_ctx == NULL) {
          img_convert_ctx = sws_getContext(
              codec_context->width,
              codec_context->height,
              codec_context->pix_fmt,
              codec_context->width,
              codec_context->height,
              dst_pix_fmt,
              SWS_BICUBIC,
              NULL,
              NULL,
              NULL);

          ASSERT(img_convert_ctx, "cannot initialize the conversion context");
        }
        sws_scale(
            img_convert_ctx,
            picture->data,
            picture->linesize,
            0,
            codec_context->height,
            pict.data,
            pict.linesize);

        SDL_UnlockYUVOverlay(bmp);

        rect.x = 0;
        rect.y = 0;
        rect.w = codec_context->width;
        rect.h = codec_context->height;
        SDL_DisplayYUVOverlay(bmp, &rect);
      }
    }

    av_free_packet(&packet);

    SDL_PollEvent(&event);
    switch(event.type) {
      case SDL_QUIT:
        SDL_Quit();
        exit(0);
        break;
      default:
        break;
    }
  }

  av_free(picture_rgb);
  av_free(picture);

  avcodec_close(codec_context);

  av_close_input_file(format_context);
}

int main(int argc, char **argv)
{
  av_register_all(); // must be called before using avcodec lib

  const char * filename = "data/test.avi";

  if (argc <= 1) {

    test_encode_stream(filename);

  } else {

    filename = argv[1];
  }

  test_decode_stream(filename);

  return 0;
}

