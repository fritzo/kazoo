
/** python extension modules: _kazoo, _transforms, _models

  TODO move constructor validation into C++ constructors
  TODO figure out how to wrap Streaming::stuff
  TODO refactor: boilerplate python c-api stuff in .h; implemention in .cpp
*/

#ifndef KAZOO_KAZOO_H
#define KAZOO_KAZOO_H

#include "common.h"
#include "audio.h"
#include "animate.h"
#include "vectors.h"
#include "splines.h"
#include "spectrogram.h"
#include "reassign.h"
#include "synchrony.h"
#include "transforms.h"
#include "psycho.h"
#include "images.h"
#include <Python.h>

// see http://docs.python.org/extending/

#define MAX_EXPONENT      (24)
#define MIN_EXPONENT      (1)
#define MAX_SIZE          (1<<MAX_EXPONENT)
#define DEFAULT_EXPONENT  (10)
#define DEFAULT_MIN_FREQ  (12e0)
#define DEFAULT_MAX_FREQ  (18e3)

//----( python c api stuff )--------------------------------------------------

#define BOILERPLATE_OBJECT(name) \
  typedef struct { \
    PyObject_HEAD \
    name * object; \
  } name ## Object; \
  \
  static PyObject * name ## _new ( \
      PyTypeObject *type, \
      PyObject *args, \
      PyObject* kwds); \
  \
  static int name ## _init ( \
      name ## Object *self, \
      PyObject* args, \
      PyObject* kwds); \
  \
  static void name ## _dealloc (name ## Object* self);

#define BOILERPLATE_TYPE(module, name) \
  static PyTypeObject name ## Type = { \
    PyObject_HEAD_INIT(NULL) \
    0,                                            /* ob_size */ \
    module "." # name,                            /* tp_name */ \
    sizeof(name ## Object),                       /* tp_basicsize */ \
    0,                                            /* tp_itemsize */ \
    (destructor) name ## _dealloc,                /* tp_dealloc */ \
    0,                                            /* tp_print */ \
    0,                                            /* tp_getattr */ \
    0,                                            /* tp_setattr */ \
    0,                                            /* tp_compare */ \
    0,                                            /* tp_repr */ \
    0,                                            /* tp_as_number */ \
    0,                                            /* tp_as_sequence */ \
    0,                                            /* tp_as_mapping */ \
    0,                                            /* tp_hash */ \
    0,                                            /* tp_call */ \
    0,                                            /* tp_str */ \
    0,                                            /* tp_getattro */ \
    0,                                            /* tp_setattro */ \
    0,                                            /* tp_as_buffer */ \
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     /* tp_flags */ \
    name ## _doc,                                 /* tp_doc */ \
    0,                                            /* tp_traverse */ \
    0,		                                        /* tp_clear */ \
    0,		                                        /* tp_richcompare */ \
    0,		                                        /* tp_weaklistoffset */ \
    0,		                                        /* tp_iter */ \
    0,		                                        /* tp_iternext */ \
    name ## _methods,                             /* tp_methods */ \
    0,                                            /* tp_members */ \
    name ##_getset,                               /* tp_getset */ \
    0,                                            /* tp_base */ \
    0,                                            /* tp_dict */ \
    0,                                            /* tp_descr_get */ \
    0,                                            /* tp_descr_set */ \
    0,                                            /* tp_dictoffset */ \
    (initproc) name ## _init,                     /* tp_init */ \
    0,                                            /* tp_alloc */ \
    name ## _new,                                 /* tp_new */ \
  };

#define NOARGS_SHAPE "\nWithout arguments, returns input,output shapes."

//----( audio object objects )------------------------------------------------

const char* Audio_doc =
"Audio(size = 2**DEFAULT_EXPONENT,\n"
"      sample_rate = DEFAULT_SAMPLE_RATE,\n"
"      reading = True,\n"
"      writing = True) -> object\n\n"
"Two-way audio object objects.";

BOILERPLATE_OBJECT(Audio)

PyObject* Audio_size (AudioObject* self);
PyObject* Audio_rate (AudioObject* self);

PyObject* Audio_reading (AudioObject* self);
PyObject* Audio_writing (AudioObject* self);

PyObject* Audio_start (AudioObject* self);
PyObject* Audio_stop (AudioObject* self);

PyObject* Audio_read (AudioObject* self, PyObject* args);
PyObject* Audio_write (AudioObject* self, PyObject* args);

static PyGetSetDef Audio_getset[] = {
    {"size", (getter)Audio_size, NULL,
     "window size in frames/buffer", NULL
    },
    {"rate", (getter)Audio_rate, NULL,
     "sampling rate in 1/sec", NULL
    },
    {"am_reading", (getter)Audio_reading, NULL,
     "whether stream is inputting audio", NULL
    },
    {"am_writing", (getter)Audio_writing, NULL,
     "whether stream is outputting audio", NULL
    },
    {NULL}
};

static PyMethodDef Audio_methods[] = {
    {"start", (PyCFunction)Audio_start, METH_NOARGS,
     "start()\n\n"
     "Starts audio object."
    },
    {"stop", (PyCFunction)Audio_stop, METH_NOARGS,
     "stop()\n\n"
     "Stops audio object."
    },
    {"read", (PyCFunction)Audio_read, METH_VARARGS,
     "read(time_data)\n\n"
     "Reads window from standard audio input."
     NOARGS_SHAPE
    },
    {"write", (PyCFunction)Audio_write, METH_VARARGS,
     "write(time_data)\n\n"
     "Writes window to standard audio output."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Audio)

//----( SDL screen objects )--------------------------------------------------

const char* Screen_doc =
"Screen(width, height) or Screen() -> object\n\n"
"Screen object for raster animation with specified size or fullscreen.";

BOILERPLATE_OBJECT(Screen)

PyObject* Screen_width (ScreenObject* self);
PyObject* Screen_height (ScreenObject* self);
PyObject* Screen_size (ScreenObject* self);

PyObject* Screen_draw_vh (ScreenObject* self, PyObject * args);
PyObject* Screen_draw_hv (ScreenObject* self, PyObject * args);
PyObject* Screen_vertical_sweep (ScreenObject* self, PyObject * args);

static PyGetSetDef Screen_getset[] = {
    {"width", (getter)Screen_width, NULL,
     "window width in pixels", NULL
    },
    {"height", (getter)Screen_height, NULL,
     "window height in pixels", NULL
    },
    {"size", (getter)Screen_size, NULL,
     "window size = width x height in pixels", NULL
    },
    {NULL}
};

static PyMethodDef Screen_methods[] = {
    {"draw_vh", (PyCFunction)Screen_draw_vh, METH_VARARGS,
     "draw_vh(array)\n\n"
     "Plots 2D array[y,x] to screen."
     NOARGS_SHAPE
    },
    {"draw_hv", (PyCFunction)Screen_draw_hv, METH_VARARGS,
     "draw_hv(array)\n\n"
     "Plots 2D array[x,y] to screen."
     NOARGS_SHAPE
    },
    {"vertical_sweep", (PyCFunction)Screen_vertical_sweep, METH_VARARGS,
     "vertical_sweep(array)\n\n"
     "Plots vertical band of raster data."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Screen)

//----( spectrogram objects )-------------------------------------------------

const char* Spectrogram_doc =
"Spectrogram(exponent = DEFAULT_EXPONENT) -> object\n\n"
"Spectrogram objects.";

BOILERPLATE_OBJECT(Spectrogram)

PyObject* Spectrogram_size (SpectrogramObject* self);
PyObject* Spectrogram_weights (SpectrogramObject* self);

PyObject* Spectrogram_transform_fwd (SpectrogramObject* self, PyObject* args);
PyObject* Spectrogram_transform_bwd (SpectrogramObject* self, PyObject* args);

static PyGetSetDef Spectrogram_getset[] = {
    {"size", (getter)Spectrogram_size, NULL,
     "window size", NULL
    },
    {"weights", (getter)Spectrogram_weights, NULL,
     "array of windowing weights", NULL
    },
    {NULL}
};

static PyMethodDef Spectrogram_methods[] = {
    {"transform_fwd", (PyCFunction)Spectrogram_transform_fwd, METH_VARARGS,
     "transform_fwd(time_value, freq_energy)\n\n"
     "Transforms a window from time --> frequency domain."
     NOARGS_SHAPE
    },
    {"transform_bwd", (PyCFunction)Spectrogram_transform_bwd, METH_VARARGS,
     "transform_bwd(freq_energy, time_value)\n\n"
     "Transforms a window from frequency --> time domain."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Spectrogram)

//----( supersampled reassigned spectrogram objects )-------------------------

const char* Supergram_doc =
"Supergram(size_exponent = DEFAULT_EXPONENT,\n"
"          time_exponent = 2,\n"
"          freq_exponent = time_exponent,\n"
"          sample_rate = DEFAULT_SAMPLE_RATE) -> object\n\n"
"Supersampled reassigned spectrogram objects.";

BOILERPLATE_OBJECT(Supergram)

PyObject* Supergram_size       (SupergramObject* self);
PyObject* Supergram_time_factor(SupergramObject* self);
PyObject* Supergram_freq_factor(SupergramObject* self);
PyObject* Supergram_small_size (SupergramObject* self);
PyObject* Supergram_super_size (SupergramObject* self);
PyObject* Supergram_sample_rate(SupergramObject* self);
PyObject* Supergram_weights    (SupergramObject* self);
PyObject* Supergram_synth      (SupergramObject* self);

PyObject* Supergram_transform_fwd (SupergramObject* self, PyObject* args);
PyObject* Supergram_transform_bwd (SupergramObject* self, PyObject* args);

PyObject* Supergram_freq_scale (SupergramObject* self, PyObject* args);
PyObject* Supergram_pitch_scale (SupergramObject* self, PyObject* args);

static PyGetSetDef Supergram_getset[] = {
    {"size", (getter)Supergram_size, NULL,
     "fft window size", NULL
    },
    {"time_factor", (getter)Supergram_time_factor, NULL,
     "time supersampling factor", NULL
    },
    {"freq_factor", (getter)Supergram_freq_factor, NULL,
     "frequency supersampling factor", NULL
    },
    {"small_size", (getter)Supergram_small_size, NULL,
     "linear spectrum size", NULL
    },
    {"super_size", (getter)Supergram_super_size, NULL,
     "super spectrum size", NULL
    },
    {"sample_rate", (getter)Supergram_sample_rate, NULL,
     "sample rate", NULL
    },
    {"weights", (getter)Supergram_weights, NULL,
     "array of analysis windowing weights", NULL
    },
    {"synth", (getter)Supergram_synth, NULL,
     "array of synthesis windowing weights", NULL
    },
    {NULL}
};

static PyMethodDef Supergram_methods[] = {
    {"transform_fwd", (PyCFunction)Supergram_transform_fwd, METH_VARARGS,
     "transform_fwd(time_value, freq_energy)\n\n"
     "Transforms a window from time --> frequency domain."
     NOARGS_SHAPE
    },
    {"transform_bwd", (PyCFunction)Supergram_transform_bwd, METH_VARARGS,
     "transform_bwd(freq_energy, time_value)\n\n"
     "Transforms a window from frequency --> time domain."
     NOARGS_SHAPE
    },
    {"freq_scale", (PyCFunction)Supergram_freq_scale, METH_VARARGS,
     "freq_scale(size_out, max_freq_hz = DEFAULT_MAX_FREQ) -> Spline\n\n"
     "Creates a Splin1D to transform superspectrum to frequency scale."
    },
    {"pitch_scale", (PyCFunction)Supergram_pitch_scale, METH_VARARGS,
     "pitch_scale(size_out, min_freq_hz = DEFAULT_MIN_FREQ,\n"
     "                      max_freq_hz = DEFAULT_MAX_FREQ) -> Spline\n\n"
     "Creates a Splin1D to transform superspectrum to log(frequency) scale."
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Supergram)

//----( phasogram objects )---------------------------------------------------

using Synchronized::Phasogram;

const char* Phasogram_doc =
"Phasogram(bank_size,\n"
"          block_size,\n"
"          freq0,\n"
"          freq1,\n"
"          acuity,\n"
"          strength) -> object\n\n"
"Bank of phasors.";

BOILERPLATE_OBJECT(Phasogram)

PyObject* Phasogram_size_in  (PhasogramObject* self);
PyObject* Phasogram_size_out (PhasogramObject* self);

PyObject* Phasogram_transform (PhasogramObject* self, PyObject* args);

static PyGetSetDef Phasogram_getset[] = {
    {"size_in", (getter)Phasogram_size_in, NULL,
     "size of oscillator bank, for control inputs", NULL
    },
    {"size_out", (getter)Phasogram_size_out, NULL,
     "block size of output duration", NULL
    },
    {NULL}
};

static PyMethodDef Phasogram_methods[] = {
    {"transform", (PyCFunction)Phasogram_transform, METH_VARARGS,
     "transform(mass_in, amplitude_in, sound_out)\n\n"
     "Integrates sum of coupled oscillators through time."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Phasogram)

//----( pitchgram objects )---------------------------------------------------

using Synchronized::Pitchgram;

const char* Pitchgram_doc =
"Pitchgram(block_size,\n"
"          bank_size,\n"
"          freq0 = DEFAULT_MIN_FREQ * DEFAULT_SAMPLE_RATE,\n"
"          freq1 = DEFAULT_MAX_FREQ * DEFAULT_SAMPLE_RATE) -> object\n\n"
"Bank of log-spaced Fourier oscillators.";

BOILERPLATE_OBJECT(Pitchgram)

PyObject* Pitchgram_size_in  (PitchgramObject* self);
PyObject* Pitchgram_size_out (PitchgramObject* self);

PyObject* Pitchgram_transform (PitchgramObject* self, PyObject* args);

static PyGetSetDef Pitchgram_getset[] = {
    {"size_in", (getter)Pitchgram_size_in, NULL,
     "block size of time-domain input signal", NULL
    },
    {"size_out", (getter)Pitchgram_size_out, NULL,
     "size of oscillator bank", NULL
    },
    {NULL}
};

static PyMethodDef Pitchgram_methods[] = {
    {"transform", (PyCFunction)Pitchgram_transform, METH_VARARGS,
     "transform(time_in, pitch_out)\n\n"
     "Fourier transforms a signal with log-spaced frequency bins."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Pitchgram)

//----( multiscale splitter )-------------------------------------------------

const char* MultiScale_doc =
"MultiScale(super_to_fst, super_to_snd) -> object\n\n"
"Multiple scale splitter/fuser objects.";

BOILERPLATE_OBJECT(MultiScale)

PyObject* MultiScale_size_super (MultiScaleObject* self);
PyObject* MultiScale_size_fst (MultiScaleObject* self);
PyObject* MultiScale_size_snd (MultiScaleObject* self);

PyObject* MultiScale_transform_fwd (MultiScaleObject* self, PyObject* args);
PyObject* MultiScale_transform_bwd (MultiScaleObject* self, PyObject* args);

static PyGetSetDef MultiScale_getset[] = {
    {"size_super", (getter)MultiScale_size_super, NULL,
     "size of input channel", NULL
    },
    {"size_fst", (getter)MultiScale_size_fst, NULL,
     "size of first output channel", NULL
    },
    {"size_snd", (getter)MultiScale_size_snd, NULL,
     "size of second output channel", NULL
    },
    {NULL}
};

static PyMethodDef MultiScale_methods[] = {
    {"transform_fwd", (PyCFunction)MultiScale_transform_fwd, METH_VARARGS,
     "transform_fwd(super_in, (fst_out, snd_out))\n\n"
     "Splits super channel into fst+snd channels."
     NOARGS_SHAPE
    },
    {"transform_bwd", (PyCFunction)MultiScale_transform_bwd, METH_VARARGS,
     "transform_bwd((fst_io, snd_io), super_out)\n\n"
     "Fuses fst+snd channels into super channel."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", MultiScale)

//----( highpass/lowpass splitter )-------------------------------------------

const char* HiLoSplitter_doc =
"HiLoSplitter(size, size_lowpass) -> object\n\n"
"highpass/lowpass splitter objects.";

BOILERPLATE_OBJECT(HiLoSplitter)

PyObject* HiLoSplitter_size (HiLoSplitterObject* self);
PyObject* HiLoSplitter_size_lowpass (HiLoSplitterObject* self);

PyObject* HiLoSplitter_transform_fwd (HiLoSplitterObject* self, PyObject* args);
PyObject* HiLoSplitter_transform_bwd (HiLoSplitterObject* self, PyObject* args);

static PyGetSetDef HiLoSplitter_getset[] = {
    {"size", (getter)HiLoSplitter_size, NULL,
     "size of input channel = size of highpass output channel", NULL
    },
    {"size_lowpass", (getter)HiLoSplitter_size_lowpass, NULL,
     "size of lowpass output channel", NULL
    },
    {NULL}
};

static PyMethodDef HiLoSplitter_methods[] = {
    {"transform_fwd", (PyCFunction)HiLoSplitter_transform_fwd, METH_VARARGS,
     "transform_fwd(full_in, (highpass_out, lowpass_out))\n\n"
     "Splits full channel into hihg+low channels."
     NOARGS_SHAPE
    },
    {"transform_bwd", (PyCFunction)HiLoSplitter_transform_bwd, METH_VARARGS,
     "transform_bwd((highpass_in, lowpass_in), full_out)\n\n"
     "Fuses high+low channels into full channel."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", HiLoSplitter)

//----( shepard scale )-------------------------------------------------------

const char* Shepard_doc =
"Shepard(size_in, min_freq_hz = DEFAULT_MIN_FREQ,\n"
"                 max_freq_hz = DEFAULT_MAX_FREQ) -> object\n\n"
"Shepard 12-tone scale objects.";

BOILERPLATE_OBJECT(Shepard)

PyObject* Shepard_size_in (ShepardObject* self);
PyObject* Shepard_size_out (ShepardObject* self);

PyObject* Shepard_transform_fwd (ShepardObject* self, PyObject* args);
PyObject* Shepard_transform_bwd (ShepardObject* self, PyObject* args);

static PyGetSetDef Shepard_getset[] = {
    {"size_in", (getter)Shepard_size_in, NULL,
     "size of input pitch scale", NULL
    },
    {"size_out", (getter)Shepard_size_out, NULL,
     "number of tones in output scale (typically 12)", NULL
    },
    {NULL}
};

static PyMethodDef Shepard_methods[] = {
    {"transform_fwd", (PyCFunction)Shepard_transform_fwd, METH_VARARGS,
     "transform_fwd(pitch_in, tone_out)\n\n"
     "Transforms pitch to tone-mod-octave, while learning tone alignment."
     NOARGS_SHAPE
    },
    {"transform_bwd", (PyCFunction)Shepard_transform_bwd, METH_VARARGS,
     "transform_bwd(tone_in, pitch_out)\n\n"
     "Transforms from tone-mod-octave to pitch."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Shepard)

//----( loudness transform )--------------------------------------------------

using Psycho::Loudness;

const char* Loudness_doc =
"Loudness(size, frame_rate) -> object\n\n"
"Energy <--> Loudness conversion objects.";

BOILERPLATE_OBJECT(Loudness)

PyObject* Loudness_size (LoudnessObject* self);
PyObject* Loudness_frame_rate (LoudnessObject* self);
PyObject* Loudness_time_scale (LoudnessObject* self);
PyObject* Loudness_ss_factor (LoudnessObject* self);

PyObject* Loudness_transform_fwd (LoudnessObject* self, PyObject* args);
PyObject* Loudness_transform_bwd (LoudnessObject* self, PyObject* args);

static PyGetSetDef Loudness_getset[] = {
    {"size", (getter)Loudness_size, NULL,
     "size of transformed spectra", NULL
    },
    {"frame_rate", (getter)Loudness_frame_rate, NULL,
     "frame rate in Hz", NULL
    },
    {"time_scale", (getter)Loudness_time_scale, NULL,
     "highpass filter time scale", NULL
    },
    {"ss_factor", (getter)Loudness_ss_factor, NULL,
     "discount factor for steady state signals", NULL
    },
    {NULL}
};

static PyMethodDef Loudness_methods[] = {
    {"transform_fwd", (PyCFunction)Loudness_transform_fwd, METH_VARARGS,
     "transform_fwd((energy_in, d_freq_in), tone_out)\n\n"
     "Transforms from frequency --> log(frequency)."
     NOARGS_SHAPE
    },
    {"transform_bwd", (PyCFunction)Loudness_transform_bwd, METH_VARARGS,
     "transform_bwd(tone_in, (energy_out, d_freq_out))\n\n"
     "Transforms from log(frequency) --> frequency."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Loudness)

//----( sharpener )-----------------------------------------------------------

const char* Sharpener_doc =
"Sharpener(size) -> object\n\n"
"Spectrum sharpening transform objects.";

BOILERPLATE_OBJECT(Sharpener)

PyObject* Sharpener_size (SharpenerObject* self);

PyObject* Sharpener_transform (SharpenerObject* self, PyObject* args);

static PyGetSetDef Sharpener_getset[] = {
    {"size", (getter)Sharpener_size, NULL,
     " size of transformed spectra", NULL
    },
    {NULL}
};

static PyMethodDef Sharpener_methods[] = {
    {"transform", (PyCFunction)Sharpener_transform, METH_VARARGS,
     "transform(freq_in, freq_out)\n\n"
     "Sharpens spectrum."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Sharpener)

//----( octave shift )--------------------------------------------------------

const char* OctaveLower_doc =
"OctaveLower(size) -> object\n\n"
"Pitch shifting transform objects.";

BOILERPLATE_OBJECT(OctaveLower)

PyObject* OctaveLower_size (OctaveLowerObject* self);

PyObject* OctaveLower_transform (OctaveLowerObject* self, PyObject* args);

static PyGetSetDef OctaveLower_getset[] = {
    {"size", (getter)OctaveLower_size, NULL,
     "size of transformed spectra", NULL
    },
    {NULL}
};

static PyMethodDef OctaveLower_methods[] = {
    {"transform", (PyCFunction)OctaveLower_transform, METH_VARARGS,
     "transform(sound_in, sound_out)\n\n"
     "Shifts pitch down an octave."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", OctaveLower)

//----( pitch shift )---------------------------------------------------------
/* OLD
const char* PitchShift_doc =
"PitchShift(size_exponent, factor_exponent, halftone_shift) -> object\n\n"
"Pitch shifting transform objects.";

BOILERPLATE_OBJECT(PitchShift)

PyObject* PitchShift_size (PitchShiftObject* self);
PyObject* PitchShift_factor (PitchShiftObject* self);

PyObject* PitchShift_transform (PitchShiftObject* self, PyObject* args);

static PyGetSetDef PitchShift_getset[] = {
    {"size", (getter)PitchShift_size, NULL,
     "size of input/output sound", NULL
    },
    {"factor", (getter)PitchShift_size, NULL,
     "freqency scaling factor", NULL
    },
    {NULL}
};

static PyMethodDef PitchShift_methods[] = {
    {"transform", (PyCFunction)PitchShift_transform, METH_VARARGS,
     "transform(sound_in, sound_out)\n\n"
     "Shifts pitch by specified amount."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", PitchShift)
*/

//----( melodigram objects )--------------------------------------------------

const char* Melodigram_doc =
"Melodigram(exponent, num_filters, frame_rate) -> object\n\n"
"Melodigram objects.";

BOILERPLATE_OBJECT(Melodigram)

PyObject* Melodigram_size (MelodigramObject* self);
PyObject* Melodigram_size_corr (MelodigramObject* self);
PyObject* Melodigram_num_filters (MelodigramObject* self);
PyObject* Melodigram_size_out (MelodigramObject* self);

PyObject* Melodigram_transform_fwd (MelodigramObject* self, PyObject* args);
PyObject* Melodigram_transform_bwd (MelodigramObject* self, PyObject* args);

static PyGetSetDef Melodigram_getset[] = {
    {"size", (getter)Melodigram_size, NULL,
     "input size", NULL
    },
    {"size_corr", (getter)Melodigram_size_corr, NULL,
     "size of each output correlation filter", NULL
    },
    {"num_filters", (getter)Melodigram_num_filters, NULL,
     "number of Laplace transform filters", NULL
    },
    {"size_out", (getter)Melodigram_size_out, NULL,
     "total output size = size_corr x num_filters", NULL
    },
    {NULL}
};

static PyMethodDef Melodigram_methods[] = {
    {"transform_fwd", (PyCFunction)Melodigram_transform_fwd, METH_VARARGS,
     "transform_fwd(pitch_in, corr_out)\n\n"
     "Computes melodigram transform."
     NOARGS_SHAPE
    },
    {"transform_bwd", (PyCFunction)Melodigram_transform_bwd, METH_VARARGS,
     "transform_bwd((prev_pitch_in, corr_in), pitch_out)\n\n"
     "Inverts melodigram transform."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Melodigram)

//----( rhythmgram objects )--------------------------------------------------

const char* Rhythmgram_doc =
"Rhythmgram(size_in, size_factor) -> object\n\n"
"Rhythmgram objects.";

BOILERPLATE_OBJECT(Rhythmgram)

PyObject* Rhythmgram_size_in (RhythmgramObject* self);
PyObject* Rhythmgram_size_factor (RhythmgramObject* self);
PyObject* Rhythmgram_size_out (RhythmgramObject* self);

PyObject* Rhythmgram_transform_fwd (RhythmgramObject* self, PyObject* args);
PyObject* Rhythmgram_transform_bwd (RhythmgramObject* self, PyObject* args);

static PyGetSetDef Rhythmgram_getset[] = {
    {"size_in", (getter)Rhythmgram_size_in, NULL,
     "input size", NULL
    },
    {"size_factor", (getter)Rhythmgram_size_factor, NULL,
     "number of channels per input", NULL
    },
    {"size_out", (getter)Rhythmgram_size_out, NULL,
     "output size = size_in x size_factor", NULL
    },
    {NULL}
};

static PyMethodDef Rhythmgram_methods[] = {
    {"transform_fwd", (PyCFunction)Rhythmgram_transform_fwd, METH_VARARGS,
     "transform_fwd(f_in, ff_out)\n\n"
     "Computes autocorrelation."
     NOARGS_SHAPE
    },
    {"transform_bwd", (PyCFunction)Rhythmgram_transform_bwd, METH_VARARGS,
     "transform_bwd((ff_in, f_in), df_out)\n\n"
     "Inverts autocorrelation."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Rhythmgram)

//----( correlogram objects )-------------------------------------------------

const char* Correlogram_doc =
"Correlogram(size, decay_factor) -> object\n\n"
"Frequency <--> Time-integragted Correlation conversion.";

BOILERPLATE_OBJECT(Correlogram)

PyObject* Correlogram_size_in (CorrelogramObject* self);
PyObject* Correlogram_size_out (CorrelogramObject* self);
PyObject* Correlogram_decay_factor (CorrelogramObject* self);

PyObject* Correlogram_transform_fwd (CorrelogramObject* self, PyObject* args);
PyObject* Correlogram_transform_bwd (CorrelogramObject* self, PyObject* args);

static PyGetSetDef Correlogram_getset[] = {
    {"size_in", (getter)Correlogram_size_in, NULL,
     "input size", NULL
    },
    {"size_out", (getter)Correlogram_size_out, NULL,
     "output size", NULL
    },
    {"decay_factor", (getter)Correlogram_decay_factor, NULL,
     "history decay factor", NULL
    },
    {NULL}
};

static PyMethodDef Correlogram_methods[] = {
    {"transform_fwd", (PyCFunction)Correlogram_transform_fwd, METH_VARARGS,
     "transform_fwd((energy_in, d_freq_in), tone_out)\n\n"
     "Transforms from frequency --> correlation."
     NOARGS_SHAPE
    },
    {"transform_bwd", (PyCFunction)Correlogram_transform_bwd, METH_VARARGS,
     "transform_bwd(tone_in, (energy_out, d_freq_out))\n\n"
     "Transforms from correlation --> frequency."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Correlogram)

//----( logarithmic history )-------------------------------------------------

using Psycho::History;

const char* History_doc =
"History(size, length, density) -> object\n\n"
"Logarithmic history objects.";

BOILERPLATE_OBJECT(History)

PyObject* History_size (HistoryObject* self);
PyObject* History_length (HistoryObject* self);
PyObject* History_density (HistoryObject* self);
PyObject* History_size_in (HistoryObject* self);
PyObject* History_size_out (HistoryObject* self);
PyObject* History_full (HistoryObject* self);
PyObject* History_shape (HistoryObject* self);

PyObject* History_add (HistoryObject* self, PyObject* args);
PyObject* History_get (HistoryObject* self, PyObject* args);
PyObject* History_transform (HistoryObject* self, PyObject* args);

static PyGetSetDef History_getset[] = {
    {"size", (getter)History_size, NULL,
     "channel size.", NULL
    },
    {"length", (getter)History_length, NULL,
     "history duration.", NULL
    },
    {"density", (getter)History_density, NULL,
     "history density, or time delay until half resolution.", NULL
    },
    {"size_in", (getter)History_size_in, NULL,
     "input size = size.", NULL
    },
    {"size_out", (getter)History_size_out, NULL,
     "total output size = size x length.", NULL
    },
    {"full", (getter)History_full, NULL,
     "whether history is fully populated.", NULL
    },
    {"shape", (getter)History_shape, NULL,
     "tuple of (length, size).", NULL
    },
    {NULL}
};

static PyMethodDef History_methods[] = {
    {"add", (PyCFunction)History_add, METH_VARARGS,
     "add(frame_in)\n\n"
     "Adds frame to history."
    },
    {"get", (PyCFunction)History_get, METH_VARARGS,
     "get(history_out)\n\n"
     "Retrieves logarithmic history of frames."
    },
    {"transform", (PyCFunction)History_transform, METH_VARARGS,
     "transform(frame_in, history_out)\n\n"
     "Adds a frame, then gets history."
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", History)

//----( 1d spline objects )---------------------------------------------------

const char* Spline_doc =
"Spline(size_in, size_out, fun = identity) -> object\n\n"
"Spline objects.\n\n"
"fun should be a size_in vector mapping to the real interval [0,1].\n"
"If no function is given, the identity mapping is assumed.";

BOILERPLATE_OBJECT(Spline)

PyObject* Spline_swap (SplineObject* self, PyObject* args);

PyObject* Spline_size_in (SplineObject* self);
PyObject* Spline_size_out (SplineObject* self);

PyObject* Spline_transform_fwd (SplineObject* self, PyObject* args);
PyObject* Spline_transform_bwd (SplineObject* self, PyObject* args);

static PyGetSetDef Spline_getset[] = {
    {"size_in", (getter)Spline_size_in, NULL,
     "input size", NULL
    },
    {"size_out", (getter)Spline_size_out, NULL,
     "output size", NULL
    },
    {NULL}
};

static PyMethodDef Spline_methods[] = {
    {"swap", (PyCFunction)Spline_swap, METH_VARARGS,
     "swap(other)\n\n"
     "Swaps data with other Spline."
    },
    {"transform_fwd", (PyCFunction)Spline_transform_fwd, METH_VARARGS,
     "transform_fwd(e_dom, e_rng)\n\n"
     "Transforms energy from dom to rng coordinates."
     NOARGS_SHAPE
    },
    {"transform_bwd", (PyCFunction)Spline_transform_bwd, METH_VARARGS,
     "transform_bwd(e_rng, e_dom)\n\n"
     "Transforms energy from rng to dom coordinates."
     NOARGS_SHAPE
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Spline)

//----( 2d spline objects )---------------------------------------------------

const char* Spline2DSeparable_doc =
"Spline2DSeparable(size_in1, size_in2, size_out1, size_out2) -> object\n\n"
"Spline2DSeparable objects.";

BOILERPLATE_OBJECT(Spline2DSeparable)

PyObject* Spline2DSeparable_shape (Spline2DSeparableObject* self);

PyObject* Spline2DSeparable_transform_fwd (Spline2DSeparableObject* self, PyObject* args);
PyObject* Spline2DSeparable_transform_bwd (Spline2DSeparableObject* self, PyObject* args);

static PyGetSetDef Spline2DSeparable_getset[] = {
    {"shape", (getter)Spline2DSeparable_shape, NULL,
     "tuple of (size_in1, size_in2, size_out1, size_out2).", NULL
    },
    {NULL}
};

static PyMethodDef Spline2DSeparable_methods[] = {
    {"transform_fwd", (PyCFunction)Spline2DSeparable_transform_fwd, METH_VARARGS,
     "transform_fwd(e_dom, e_rng)\n\n"
     "Scales a larger image to a smaller image."
    },
    {"transform_bwd", (PyCFunction)Spline2DSeparable_transform_bwd, METH_VARARGS,
     "transform_bwd(e_rng, e_dom)\n\n"
     "Scales a smaller image to a larger image."
    },
    {NULL}
};

BOILERPLATE_TYPE("_transforms", Spline2DSeparable)

//----( misc functions )------------------------------------------------------

PyObject * new_Vector<float> (PyObject * self, PyObject * args);
PyObject * new_Vector<complex> (PyObject * self, PyObject * args);
PyObject * hdr_real_color (PyObject * self, PyObject * args);

//----( kazoo module )--------------------------------------------------------

static PyMethodDef kazoo_methods[] = {
  { "Vector<float>", new_Vector<float>, METH_VARARGS,
    "Vector<float>(size0 [,size1, size2, size3])\n\n"
    "Creates a memory-aligned array of float32 values initialized to zero."
    "WARNING: the array memory is never freed."},
  { "Vector<complex>", new_Vector<complex>, METH_VARARGS,
    "Vector<complex>(size0 [,size1, size2, size3])\n\n"
    "Creates a memory-aligned array of complex64 values initialized to zero."
    "WARNING: the array memory is never freed."},
  { "hdr_real_color", hdr_real_color, METH_VARARGS,
    "hdr_real_color(width, height, r,g,b)\n\n"
    "Applies fake hdr transform to real-valued image."},
  { NULL, NULL, 0, NULL }
};

#endif // KAZOO_KAZOO_H

