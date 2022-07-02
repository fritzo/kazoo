
#include "_kazoo.h"
#include "events.h"

// see http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#include <numpy/arrayobject.h>

// see http://web.archive.org/web/20021218064807/http://starship.python.net/crew/hinsen/NumPyExtensions.html
// see http://docs.scipy.org/doc/numpy/reference/c-api.html

//----( python c api stuff )--------------------------------------------------

#define BOILERPLATE_NEW(name) \
  static PyObject * name ## _new ( \
      PyTypeObject * type, \
      PyObject * args, \
      PyObject * kwds) \
  { \
    name ## Object * self; \
    self = (name ## Object *) type->tp_alloc(type, 0); \
    if (self == NULL) return NULL; \
 \
    self->object = NULL; \
 \
    return (PyObject *) self; \
  } \
 \
  static void name ## _dealloc (name ## Object * self) \
  { \
    if (self->object == NULL) { \
      WARN(# name " NULL on dealloc"); \
    } else { \
      delete self->object; \
      self->object = NULL; \
    } \
 \
    Py_TYPE(self)->tp_free((PyObject*)self); \
  }

#define INT_GETTER(Class,name) \
  PyObject* Class ## _ ## name (Class ## Object * self) \
  { \
    Class * object = self->object; \
    ASSERT(object != NULL, # Class " NULL"); \
    return PyLong_FromLong(object->name()); \
  }
#define FLOAT_GETTER(Class,name) \
  PyObject* Class ## _ ## name (Class ## Object * self) \
  { \
    Class * object = self->object; \
    ASSERT(object != NULL, # Class " NULL"); \
    return PyFloat_FromDouble(object->name()); \
  }

#define BOOL_GETTER(Class,name) \
  PyObject* Class ## _ ## name (Class ## Object * self) \
  { \
    Class * object = self->object; \
    ASSERT(object != NULL, # Class " NULL"); \
    return PyBool_FromLong(object->name()); \
  }

// The source for PyArg_NoArgs is marked "should not be used"
// but we have a valid use below, testing for empty argument lists.
#define PyArg_Empty(args) (PyTuple_Size(args) == 0)

//----( logging )-------------------------------------------------------------

#define LOG1(mess)
//#define LOG1(mess) LOG(mess)

#define ASSERTVAL(cond,mess) {if(!(cond)){\
  PyErr_SetString(PyExc_ValueError,mess);\
  return NULL;}}

#define ASSERT_CONSTRUCTOR(name,cond) { if(!(cond)){\
  WARN( "bad arguments to " #name "() constructor");\
  return -1;}}

#define ASSERT_REALS(array,size) {\
  ASSERTVAL(PyArray_NDIM(array) == 1, \
            QUOTE(array) " must be 1-dimensional"); \
  ASSERTVAL(PyArray_TYPE(array) == PyArray_FLOAT, \
            QUOTE(array) " must have type float32"); \
  ASSERTVAL(static_cast<size_t>(PyArray_DIM((array),0)) == (size), \
            QUOTE(array) " must have correct size"); \
  ASSERTVAL(PyArray_STRIDE((array),0) == sizeof(float), \
            QUOTE(array) " must have unit stride"); }

#define ASSERT_COMPLEXES(array,size) {\
  ASSERTVAL(PyArray_NDIM(array) == 1, \
            QUOTE(array) " must be 1-dimensional"); \
  ASSERTVAL(PyArray_TYPE(array) == PyArray_CFLOAT, \
            QUOTE(array) " must have type complex64"); \
  ASSERTVAL(static_cast<size_t>(PyArray_DIM((array),0)) == (size), \
            QUOTE(array) " must have correct size"); \
  ASSERTVAL(PyArray_STRIDE((array),0) == sizeof(complex), \
            QUOTE(array) " must have unit stride"); }

#define ASSERT_REALS2(array,size0,size1) {\
  ASSERTVAL(PyArray_NDIM(array) == 2, \
            QUOTE(array) " must be 2-dimensional"); \
  ASSERTVAL(PyArray_TYPE(array) == PyArray_FLOAT, \
            QUOTE(array) " must have type float32"); \
  ASSERTVAL(static_cast<size_t>(PyArray_DIM((array),0)) == (size0), \
            QUOTE(array) " must have correct size0"); \
  ASSERTVAL(static_cast<size_t>(PyArray_DIM((array),1)) == (size1), \
            QUOTE(array) " must have correct size1"); \
  ASSERTVAL(PyArray_STRIDE((array),0) == static_cast<int>(sizeof(float) * (size1)), \
            QUOTE(array) " must have unit stride0 / size1"); \
  ASSERTVAL(PyArray_STRIDE((array),1) == static_cast<int>(sizeof(float)), \
            QUOTE(array) " must have unit stride1"); }

//----( vector conversions )--------------------------------------------------

PyArrayObject * py_Vector_float (const Vector<float> & vector)
{
  size_t nd = 1;
  static npy_intp dims[1];
  dims[0] = vector.size;
  PyObject* array = PyArray_SimpleNew(nd, dims, PyArray_FLOAT);
  copy_float(vector, (float *)PyArray_DATA(array), vector.size);
  return (PyArrayObject *) array;
}

Vector<float> c_Vector_float (PyArrayObject * array, size_t true_size = 0)
{
  size_t nd = PyArray_NDIM(array);
  ASSERT(PyArray_TYPE(array) == PyArray_FLOAT,
         "array must have type float32");
  ASSERT(PyArray_STRIDE((array), nd-1) == sizeof(float),
         "array must have unit stride");

  size_t size = 1;
  for (size_t dim = 0; dim < nd; ++dim) {
    size *= static_cast<size_t>(PyArray_DIM((array),dim));
  }
  if (true_size) {
    ASSERT(size == true_size,
           "size mismatch: " << size << " != " << true_size);
  }

  return Vector<float>(size, (float*)PyArray_DATA(array));
}

Vector<complex> c_Vector_complex (PyArrayObject * array, size_t true_size = 0)
{
  ASSERT(PyArray_TYPE(array) == PyArray_CFLOAT,
         "array must have type complex64");
  ASSERT(PyArray_STRIDE((array),0) == sizeof(complex),
         "array must have unit stride");

  size_t size = 1;
  size_t nd = PyArray_NDIM(array);
  for (size_t dim = 0; dim < nd; ++dim) {
    size *= static_cast<size_t>(PyArray_DIM((array),dim));
  }
  if (true_size) {
    ASSERT(size == true_size,
           "size mismatch: " << size << " != " << true_size);
  }

  return Vector<complex>(size, (complex*)PyArray_DATA(array));
}

//----( kazoo module )--------------------------------------------------------

// declarations for DLL import/export
#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif

#define NEW_MODULE(c_name, py_name, methods, description) \
  static struct PyModuleDef c_name ## _struct = { \
    PyModuleDef_HEAD_INIT, py_name, description, -1, methods }; \
  PyObject* c_name = PyModule_Create(&c_name ## _struct); \
  if (c_name == NULL) return NULL;

#define ADD_OBJ_TO_MODULE(module,name,obj) { Py_INCREF( & (obj)); \
  PyModule_AddObject((module), (name), (PyObject *) & (obj)); }

PyMODINIT_FUNC init_kazoo (void)
{
  import_array();

  if (PyType_Ready(&AudioType) < 0) return NULL;
  if (PyType_Ready(&ScreenType) < 0) return NULL;
  if (PyType_Ready(&SpectrogramType) < 0) return NULL;
  if (PyType_Ready(&SupergramType) < 0) return NULL;
  if (PyType_Ready(&PhasogramType) < 0) return NULL;
  if (PyType_Ready(&PitchgramType) < 0) return NULL;
  if (PyType_Ready(&MultiScaleType) < 0) return NULL;
  if (PyType_Ready(&HiLoSplitterType) < 0) return NULL;
  if (PyType_Ready(&ShepardType) < 0) return NULL;
  if (PyType_Ready(&LoudnessType) < 0) return NULL;
  if (PyType_Ready(&SharpenerType) < 0) return NULL;
  if (PyType_Ready(&OctaveLowerType) < 0) return NULL;
  //if (PyType_Ready(&PitchShiftType) < 0) return NULL;
  if (PyType_Ready(&MelodigramType) < 0) return NULL;
  if (PyType_Ready(&RhythmgramType) < 0) return NULL;
  if (PyType_Ready(&CorrelogramType) < 0) return NULL;
  if (PyType_Ready(&HistoryType) < 0) return NULL;
  if (PyType_Ready(&SplineType) < 0) return NULL;
  if (PyType_Ready(&Spline2DSeparableType) < 0) return NULL;

  NEW_MODULE(k, "_kazoo", kazoo_methods, "Low-level array tools.");
  NEW_MODULE(t, "_transforms", NULL, "Invertible time-series transforms.");
  NEW_MODULE(m, "_models", NULL, "Partially observable time-series models.");

  ADD_OBJ_TO_MODULE(k, "transforms", t);
  ADD_OBJ_TO_MODULE(m, "models", t);

  ADD_OBJ_TO_MODULE(t, "Audio", AudioType);
  ADD_OBJ_TO_MODULE(t, "Screen", ScreenType);
  ADD_OBJ_TO_MODULE(t, "Spectrogram", SpectrogramType);
  ADD_OBJ_TO_MODULE(t, "Supergram", SupergramType);
  ADD_OBJ_TO_MODULE(t, "Phasogram", PhasogramType);
  ADD_OBJ_TO_MODULE(t, "Pitchgram", PitchgramType);
  ADD_OBJ_TO_MODULE(t, "MultiScale", MultiScaleType);
  ADD_OBJ_TO_MODULE(t, "HiLoSplitter", HiLoSplitterType);
  ADD_OBJ_TO_MODULE(t, "Shepard", ShepardType);
  ADD_OBJ_TO_MODULE(t, "Loudness", LoudnessType);
  ADD_OBJ_TO_MODULE(t, "Sharpener", SharpenerType);
  ADD_OBJ_TO_MODULE(t, "OctaveLower", OctaveLowerType);
  //ADD_OBJ_TO_MODULE(t, "PitchShift", PitchShiftType);
  ADD_OBJ_TO_MODULE(t, "Melodigram", MelodigramType);
  ADD_OBJ_TO_MODULE(t, "Rhythmgram", RhythmgramType);
  ADD_OBJ_TO_MODULE(t, "Correlogram", CorrelogramType);
  ADD_OBJ_TO_MODULE(t, "History", HistoryType);
  ADD_OBJ_TO_MODULE(t, "Spline", SplineType);
  ADD_OBJ_TO_MODULE(t, "Spline2DSeparable", Spline2DSeparableType);

  PyModule_AddIntConstant(t, "MIN_EXPONENT", MIN_EXPONENT);
  PyModule_AddIntConstant(t, "MAX_EXPONENT", MAX_EXPONENT);
  PyModule_AddIntConstant(t, "DEFAULT_EXPONENT", DEFAULT_EXPONENT);
  PyModule_AddIntConstant(t, "DEFAULT_SAMPLE_RATE", DEFAULT_SAMPLE_RATE);
  PyModule_AddIntConstant(t, "DEFAULT_FRAMES_PER_BUFFER", DEFAULT_FRAMES_PER_BUFFER);
  PyModule_AddIntConstant(t, "DEFAULT_MIN_FREQ", DEFAULT_MIN_FREQ);
  PyModule_AddIntConstant(t, "DEFAULT_MAX_FREQ", DEFAULT_MAX_FREQ);

  return k;
}

//----( audio object object )-------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Audio)

static int Audio_init (
    AudioObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size = 1 << DEFAULT_EXPONENT;
  int rate = DEFAULT_SAMPLE_RATE;
  int reading = true;
  int writing = true;
  static char * kwlist[] = {"size", "rate", "reading", "writing", NULL};
  ASSERT_CONSTRUCTOR(Audio,
          PyArg_ParseTupleAndKeywords(args, kwds, "|iiii", kwlist,
                                      &size,
                                      &rate,
                                      &reading,
                                      &writing));
  if (not (reading or writing)) {
    PyErr_SetString(PyExc_ValueError,
        "Audio must either read or write (or both)");
    return -1;
  }

  LOG("building "
   << (reading ? "reading-" : "")
   << (writing ? "writing-" : "")
   << "Audio with window size " << size
   << ", sampling rate " << rate << "Hz");
  ASSERT(self->object == NULL, "Audio not NULL on init");
  self->object = new Audio(size, rate, reading, writing);

  return 0;
}

//----( specific methods )----

INT_GETTER(Audio,size)
INT_GETTER(Audio,rate)
BOOL_GETTER(Audio,reading);
BOOL_GETTER(Audio,writing);

PyObject* Audio_start (AudioObject * self)
{
  Audio* object = self->object;
  ASSERT(object != NULL, "Audio NULL on start()");

  object->start();

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Audio_stop (AudioObject * self)
{
  Audio* object = self->object;
  ASSERT(object != NULL, "Audio NULL on stop()");

  object->stop();

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Audio_read (AudioObject * self, PyObject * args)
{
  Audio* object = self->object;
  ASSERT(object != NULL, "Audio NULL");

  ASSERTVAL(object->reading(), "tried to read from write-only Audio object");

  if (PyArg_Empty(args)) return Py_BuildValue("i", object->size());

  LOG1("parsing arguments");
  PyArrayObject * time_in;
  if (not PyArg_ParseTuple(args, "O!",
                           &PyArray_Type, &time_in)) return NULL;

  LOG1("checking array");
  ASSERT_COMPLEXES(time_in, object->size());

  LOG1("reading audio input");
  Vector<complex> c_time_in = c_Vector_complex(time_in);
  Py_BEGIN_ALLOW_THREADS
  object->read(c_time_in);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Audio_write (AudioObject * self, PyObject * args)
{
  Audio* object = self->object;
  ASSERT(object != NULL, "Audio NULL");

  ASSERTVAL(object->writing(), "tried to write to read-only Audio object");

  if (PyArg_Empty(args)) return Py_BuildValue("i", object->size());

  LOG1("parsing arguments");
  PyArrayObject * time_out;
  if (not PyArg_ParseTuple(args, "O!",
                           &PyArray_Type, &time_out)) return NULL;

  LOG1("checking array");
  ASSERT_COMPLEXES(time_out, object->size());

  LOG1("writing audio output");
  Vector<complex> c_time_out = c_Vector_complex(time_out);
  Py_BEGIN_ALLOW_THREADS
  object->write(c_time_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( SDL screen objects )--------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Screen)

static int Screen_init (
    ScreenObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int width = 0;
  int height = 0;
  ASSERT_CONSTRUCTOR(Screen,
          PyArg_ParseTuple(args, "|ii",
                           &width,
                           &height));

  if (width && height) {
    LOG("building Screen : " << width << " x " << height);
    ASSERT(self->object == NULL, "Screen not NULL on init");
    self->object = new Screen(Rectangle(width,height));
  } else {
    LOG("building Screen with maximum width,height");
    ASSERT(self->object == NULL, "Screen not NULL on init");
    self->object = new Screen();
  }

  return 0;
}

//----( specific methods )----

INT_GETTER(Screen,width)
INT_GETTER(Screen,height)
INT_GETTER(Screen,size)

PyObject* Screen_draw_vh (ScreenObject * self, PyObject * args)
{
  Screen* object = self->object;
  ASSERT(object != NULL, "Screen NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->height(),
                                                    object->width());

  LOG1("parsing arguments");
  PyArrayObject * data;

  if (not PyArg_ParseTuple(args, "O!",
                           &PyArray_Type, &data)) return NULL;

  LOG1("checking array");
  ASSERT_REALS2(data, object->height(), object->width());

  LOG1("drawing 2D array");
  Vector<float> c_data = c_Vector_float(data);
  Py_BEGIN_ALLOW_THREADS
  object->draw(c_data, false);
  if (key_pressed()) exit(0);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Screen_draw_hv (ScreenObject * self, PyObject * args)
{
  Screen* object = self->object;
  ASSERT(object != NULL, "Screen NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->width(),
                                                    object->height());

  LOG1("parsing arguments");
  PyArrayObject * data;

  if (not PyArg_ParseTuple(args, "O!",
                           &PyArray_Type, &data)) return NULL;

  LOG1("checking array");
  ASSERT_REALS2(data, object->width(), object->height());

  LOG1("drawing 2D array");
  Vector<float> c_data = c_Vector_float(data);
  Py_BEGIN_ALLOW_THREADS
  object->draw(c_data, true);
  if (key_pressed()) exit(0);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Screen_vertical_sweep (ScreenObject * self, PyObject * args)
{
  Screen* object = self->object;
  ASSERT(object != NULL, "Screen NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("i", object->height());

  LOG1("parsing arguments");
  PyArrayObject * data;

  if (not PyArg_ParseTuple(args, "O!",
                           &PyArray_Type, &data)) return NULL;

  LOG1("checking array");
  ASSERT_REALS(data, object->height());

  LOG1("drawing vertical sweep");
  Vector<float> c_data = c_Vector_float(data);
  Py_BEGIN_ALLOW_THREADS
  object->vertical_sweep(c_data);
  if (key_pressed()) exit(0);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( spectrogram objects )-------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Spectrogram)

static int Spectrogram_init (
    SpectrogramObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int exponent = DEFAULT_EXPONENT;
  ASSERT_CONSTRUCTOR(Spectrogram,
          PyArg_ParseTuple(args, "|i", &exponent));
  ASSERT((0 < exponent) and (exponent < MAX_EXPONENT),
         "exponent out of range: " << exponent);

  LOG("building Spectrogram with window size " << (1 << exponent));
  ASSERT(self->object == NULL, "Spectrogram not NULL on init");
  self->object = new Spectrogram(exponent);

  return 0;
}

//----( specific methods )----

INT_GETTER(Spectrogram,size)

PyObject* Spectrogram_weights (SpectrogramObject * self)
{
  Spectrogram* object = self->object;
  ASSERT(object != NULL, "Spectrogram NULL");

  PyArrayObject* result = py_Vector_float(object->weights());

  Py_INCREF(result);
  return (PyObject *) result;
}

PyObject* Spectrogram_transform_fwd (SpectrogramObject * self, PyObject * args)
{
  Spectrogram* object = self->object;
  ASSERT(object != NULL, "Spectrogram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size(),
                                                    object->size());

  LOG1("parsing arguments");
  PyArrayObject * time_in;
  PyArrayObject * freq_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &time_in,
                           &PyArray_Type, &freq_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_COMPLEXES(time_in, object->size());
  ASSERT_REALS(freq_out, object->size() / 2);

  LOG1("running fft");
  Vector<complex> c_time_in = c_Vector_complex(time_in);
  Vector<float> c_freq_out = c_Vector_float(freq_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_fwd(c_time_in,
                        c_freq_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Spectrogram_transform_bwd (SpectrogramObject * self, PyObject * args)
{
  Spectrogram* object = self->object;
  ASSERT(object != NULL, "Spectrogram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size(),
                                                    object->size());

  LOG1("parsing arguments");
  PyArrayObject * freq_in;
  PyArrayObject * time_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &freq_in,
                           &PyArray_Type, &time_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(freq_in, object->size() / 2);
  ASSERT_COMPLEXES(time_out, object->size());

  LOG1("running fft");
  Vector<float> c_freq_in = c_Vector_float(freq_in);
  Vector<complex> c_time_out = c_Vector_complex(time_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_bwd(c_freq_in,
                        c_time_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( supersampled reassigned spectrogram objects )-------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Supergram)

static int Supergram_init (
    SupergramObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size_exponent = DEFAULT_EXPONENT;
  int time_exponent = 2;
  int freq_exponent = -1;
  float sample_rate = DEFAULT_SAMPLE_RATE;
  ASSERT_CONSTRUCTOR(Supergram,
          PyArg_ParseTuple(args, "|iiif",
                           &size_exponent,
                           &time_exponent,
                           &freq_exponent,
                           &sample_rate));
  if (freq_exponent < 0) freq_exponent = time_exponent;
  ASSERT(2 <= size_exponent,
         "size_exponent is too small: " << size_exponent << " < 2");
  ASSERT(1 <= time_exponent,
         "time_exponent is too small: " << time_exponent << " < 1");
  ASSERT(1 <= freq_exponent,
         "freq_exponent is too small: " << freq_exponent << " < 1");
  ASSERT(2 <= size_exponent - time_exponent,
         "size_exponent - time_exponent is too small: "
         << (size_exponent - time_exponent) << " < 2");
  ASSERT(size_exponent + freq_exponent <= MAX_EXPONENT,
         "size_exponent + freq_exponent is too large: "
         << (size_exponent + freq_exponent) << " > " << MAX_EXPONENT);

  LOG("building Supergram : 2^(" << size_exponent << " - "
                                 << time_exponent << " + "
                                 << freq_exponent << ")");
  ASSERT(self->object == NULL, "Supergram not NULL on init");
  self->object = new Supergram(size_exponent,
                               time_exponent,
                               freq_exponent,
                               sample_rate);

  return 0;
}

//----( specific methods )----

INT_GETTER(Supergram, size)
INT_GETTER(Supergram, time_factor)
INT_GETTER(Supergram, freq_factor)
INT_GETTER(Supergram, small_size)
INT_GETTER(Supergram, super_size)
FLOAT_GETTER(Supergram, sample_rate)

PyObject* Supergram_weights (SupergramObject * self)
{
  Supergram* object = self->object;
  ASSERT(object != NULL, "Supergram NULL");

  PyArrayObject * result = py_Vector_float(object->weights());

  Py_INCREF(result);
  return (PyObject *) result;
}

PyObject* Supergram_synth (SupergramObject * self)
{
  Supergram* object = self->object;
  ASSERT(object != NULL, "Supergram NULL");

  PyArrayObject * result = py_Vector_float(object->synth());

  Py_INCREF(result);
  return (PyObject *) result;
}

PyObject* Supergram_transform_fwd (SupergramObject * self, PyObject * args)
{
  Supergram* object = self->object;
  ASSERT(object != NULL, "Supergram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->small_size(),
                                                    object->super_size());

  LOG1("parsing arguments");
  PyArrayObject * time_in;
  PyArrayObject * freq_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &time_in,
                           &PyArray_Type, &freq_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_COMPLEXES(time_in, object->small_size());
  ASSERT_REALS(freq_out, object->super_size());

  LOG1("running fft");
  Vector<complex> c_time_in = c_Vector_complex(time_in, object->small_size());
  Vector<float> c_freq_out = c_Vector_float(freq_out, object->super_size());
  Py_BEGIN_ALLOW_THREADS
  object->transform_fwd(c_time_in,
                        c_freq_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Supergram_transform_bwd (SupergramObject * self, PyObject * args)
{
  Supergram* object = self->object;
  ASSERT(object != NULL, "Supergram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->super_size(),
                                                    object->small_size());

  LOG1("parsing arguments");
  PyArrayObject * freq_in;
  PyArrayObject * time_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &freq_in,
                           &PyArray_Type, &time_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(freq_in, object->super_size());
  ASSERT_COMPLEXES(time_out, object->small_size());

  LOG1("running fft");
  Vector<float> c_freq_in = c_Vector_float(freq_in);
  Vector<complex> c_time_out = c_Vector_complex(time_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_bwd(c_freq_in,
                        c_time_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Supergram_freq_scale (SupergramObject * self, PyObject * args)
{
  Supergram* object = self->object;
  ASSERT(object != NULL, "Supergram NULL");

  LOG1("parsing arguments");
  int size_out = 0;

  if (not PyArg_ParseTuple(args, "|i",
                           &size_out)) return NULL;
  ASSERT(size_out >= 0,
         "size_out is too small: " << size_out);

  LOG1("building new Spline python object");
  SplineObject * result = (SplineObject *) Spline_new(&SplineType, NULL, NULL);
  result->object = object->new_FreqScale(size_out);

  Py_INCREF(result);
  return (PyObject *) result;
}

PyObject* Supergram_pitch_scale (SupergramObject * self, PyObject * args)
{
  Supergram* object = self->object;
  ASSERT(object != NULL, "Supergram NULL");

  LOG1("parsing arguments");
  int size_out = 0;

  if (not PyArg_ParseTuple(args, "|i",
                           &size_out)) return NULL;
  ASSERT(size_out >= 0,
         "size_out is too small: " << size_out);

  LOG1("building new Spline python object");
  SplineObject * result = (SplineObject *) Spline_new(&SplineType, NULL, NULL);
  result->object = object->new_PitchScale(size_out);

  Py_INCREF(result);
  return (PyObject *) result;
}

//----( phasogram objects )---------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Phasogram)

static int Phasogram_init (
    PhasogramObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int block_size;
  Synchronized::Bank param;
  ASSERT_CONSTRUCTOR(Phasogram,
          PyArg_ParseTuple(args, "iiffff",
                           &param.size,
                           &block_size,
                           &param.freq0,
                           &param.freq1,
                           &param.acuity,
                           &param.strength));
  ASSERT_LT(0, param.size);
  ASSERT_DIVIDES(4, param.size);
  ASSERT_LT(0, block_size);

  ASSERT_LT(0, param.freq0);
  ASSERT_LT(0, param.freq1);
  ASSERT_LT(0, param.acuity);
  ASSERT_LT(0, param.strength);

  LOG("building Phasogram(" << param.size << ", " << block_size << ", ...)");
  ASSERT(self->object == NULL, "Phasogram not NULL on init");
  self->object = new Phasogram(block_size, param);

  return 0;
}

//----( specific methods )----

INT_GETTER(Phasogram, size_in)
INT_GETTER(Phasogram, size_out)

PyObject* Phasogram_transform (PhasogramObject * self, PyObject * args)
{
  Phasogram* object = self->object;
  ASSERT(object != NULL, "Phasogram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("(ii)i",
                                              object->size_in(),
                                              object->size_in(),
                                              object->size_out());

  LOG1("parsing arguments");
  PyArrayObject * mass_in;
  PyArrayObject * amplitude_in;
  PyArrayObject * sound_out;

  if (not PyArg_ParseTuple(args, "O!O!O!",
                           &PyArray_Type, &mass_in,
                           &PyArray_Type, &amplitude_in,
                           &PyArray_Type, &sound_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(mass_in, object->size_in());
  ASSERT_REALS(amplitude_in, object->size_in());
  ASSERT_COMPLEXES(sound_out, object->size_out());

  LOG1("running transform");
  Vector<float> c_mass_in = c_Vector_float(mass_in, object->size_in());
  Vector<float> c_amplitude_in = c_Vector_float(amplitude_in, object->size_in());
  Vector<complex> c_sound_out = c_Vector_complex(sound_out, object->size_out());

  Py_BEGIN_ALLOW_THREADS
  object->transform(
      c_mass_in,
      c_amplitude_in,
      c_sound_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( pitchgram objects )---------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Pitchgram)

static int Pitchgram_init (
    PitchgramObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int block_size;
  int bank_size;
  float freq0 = DEFAULT_MIN_FREQ * DEFAULT_SAMPLE_RATE;
  float freq1 = DEFAULT_MAX_FREQ * DEFAULT_SAMPLE_RATE;
  ASSERT_CONSTRUCTOR(Pitchgram,
          PyArg_ParseTuple(args, "ii|ff",
                           &block_size,
                           &bank_size,
                           &freq0,
                           &freq1));
  ASSERT_LT(0, block_size);
  ASSERT_LT(0, bank_size);
  ASSERT_DIVIDES(4, bank_size);

  ASSERT_LT(0, freq0);
  ASSERT_LT(0, freq1);

  LOG("building Pitchgram(" << block_size << ", " << bank_size << ", ...)");
  ASSERT(self->object == NULL, "Pitchgram not NULL on init");
  self->object = new Pitchgram(block_size, bank_size, freq0, freq1);

  return 0;
}

//----( specific methods )----

INT_GETTER(Pitchgram, size_in)
INT_GETTER(Pitchgram, size_out)

PyObject* Pitchgram_transform (PitchgramObject * self, PyObject * args)
{
  Pitchgram* object = self->object;
  ASSERT(object != NULL, "Pitchgram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii",
                                              object->size_in(),
                                              object->size_out());

  LOG1("parsing arguments");
  PyArrayObject * time_in;
  PyArrayObject * pitch_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &time_in,
                           &PyArray_Type, &pitch_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_COMPLEXES(time_in, object->size_in());
  ASSERT_REALS(pitch_out, object->size_out());

  LOG1("running transform");
  Vector<complex> c_time_in = c_Vector_complex(time_in, object->size_in());
  Vector<float> c_pitch_out = c_Vector_float(pitch_out, object->size_out());

  Py_BEGIN_ALLOW_THREADS
  object->transform(
      c_time_in,
      c_pitch_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( multiscale splitter )-------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(MultiScale)

static int MultiScale_init (
    MultiScaleObject * self,
    PyObject * args,
    PyObject * kwds)
{
  PyObject * super_to_fst;
  PyObject * super_to_snd;
  ASSERT_CONSTRUCTOR(MultiScale,
          PyArg_ParseTuple(args, "O!O!",
                           &SplineType, &super_to_fst,
                           &SplineType, &super_to_snd));
  const Spline * spline_fst = ((SplineObject *)super_to_fst)->object;
  const Spline * spline_snd = ((SplineObject *)super_to_snd)->object;
  ASSERT(spline_fst->size_in() == spline_snd->size_in(),
         "MultiScale spline input sizes disagree: "
         << spline_fst->size_in() << " != " << spline_snd->size_in());

  const size_t size_super = spline_fst->size_in();
  const size_t size_fst   = spline_fst->size_out();
  const size_t size_snd   = spline_snd->size_out();
  LOG("building MultiScale : "
      << size_super << " --> " << size_fst << " + " << size_snd);

  ASSERT(self->object == NULL, "MultiScale not NULL on init");
  self->object = new MultiScale(spline_fst, spline_snd);

  // we don't want the splines to disappear
  // WARNING HACK data is never deallocated
  Py_INCREF(super_to_fst);
  Py_INCREF(super_to_snd);

  return 0;
}

//----( specific methods )----

INT_GETTER(MultiScale, size_super)
INT_GETTER(MultiScale, size_fst)
INT_GETTER(MultiScale, size_snd)

PyObject* MultiScale_transform_fwd (MultiScaleObject * self, PyObject * args)
{
  MultiScale* object = self->object;
  ASSERT(object != NULL, "MultiScale NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("i(ii)", object->size_super(),
                                                       object->size_fst(),
                                                       object->size_snd());

  LOG1("parsing arguments");
  PyArrayObject * super_in;
  PyArrayObject * fst_out;
  PyArrayObject * snd_out;

  if (not PyArg_ParseTuple(args, "O!(O!O!)",
                           &PyArray_Type, &super_in,
                           &PyArray_Type, &fst_out,
                           &PyArray_Type, &snd_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(super_in, object->size_super());
  ASSERT_REALS(fst_out, object->size_fst());
  ASSERT_REALS(snd_out, object->size_snd());

  LOG1("splitting super --> fst + snd");
  Vector<float> c_super_in = c_Vector_float(super_in);
  Vector<float> c_fst_out = c_Vector_float(fst_out);
  Vector<float> c_snd_out = c_Vector_float(snd_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_fwd(c_super_in,
                        c_fst_out,
                        c_snd_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* MultiScale_transform_bwd (MultiScaleObject * self, PyObject * args)
{
  MultiScale* object = self->object;
  ASSERT(object != NULL, "MultiScale NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("(ii)i", object->size_fst(),
                                                       object->size_snd(),
                                                       object->size_super());

  LOG("parsing arguments");
  PyArrayObject * fst_io;
  PyArrayObject * snd_io;
  PyArrayObject * super_out;

  if (not PyArg_ParseTuple(args, "(O!O!)O!",
                           &PyArray_Type, &fst_io,
                           &PyArray_Type, &snd_io,
                           &PyArray_Type, &super_out)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS(fst_io, object->size_fst());
  ASSERT_REALS(snd_io, object->size_snd());
  ASSERT_REALS(super_out, object->size_super());

  LOG("fusing fst + snd --> super");
  Vector<float> c_fst_io = c_Vector_float(fst_io);
  Vector<float> c_snd_io = c_Vector_float(snd_io);
  Vector<float> c_super_out = c_Vector_float(super_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_bwd(c_fst_io,
                        c_snd_io,
                        c_super_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( highpass/lowpass splitter )-------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(HiLoSplitter)

static int HiLoSplitter_init (
    HiLoSplitterObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size;
  int size_lowpass;
  ASSERT_CONSTRUCTOR(HiLoSplitter,
          PyArg_ParseTuple(args, "ii",
                           &size,
                           &size_lowpass));
  ASSERT((2 < size_lowpass) and (size_lowpass < size),
         "bad HiLoSplitter sizes: "
         << size << " full, " << size_lowpass << " lowpass");

  LOG("building HiLoSplitter : " << size << " full + "
                                 << size_lowpass << " lowpass");

  ASSERT(self->object == NULL, "HiLoSplitter not NULL on init");
  self->object = new HiLoSplitter(size, size_lowpass);

  return 0;
}

//----( specific methods )----

INT_GETTER(HiLoSplitter, size)
INT_GETTER(HiLoSplitter, size_lowpass)

PyObject* HiLoSplitter_transform_fwd (HiLoSplitterObject * self, PyObject * args)
{
  HiLoSplitter* object = self->object;
  ASSERT(object != NULL, "HiLoSplitter NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("i(ii)", object->size(),
                                                       object->size(),
                                                       object->size_lowpass());

  LOG1("parsing arguments");
  PyArrayObject * full_in;
  PyArrayObject * high_out;
  PyArrayObject * low_out;

  if (not PyArg_ParseTuple(args, "O!(O!O!)",
                           &PyArray_Type, &full_in,
                           &PyArray_Type, &high_out,
                           &PyArray_Type, &low_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(full_in, object->size());
  ASSERT_REALS(high_out, object->size());
  ASSERT_REALS(low_out, object->size_lowpass());

  LOG1("splitting full --> high + low");
  Vector<float> c_full_in = c_Vector_float(full_in);
  Vector<float> c_high_out = c_Vector_float(high_out);
  Vector<float> c_low_out = c_Vector_float(low_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_fwd(c_full_in,
                        c_high_out,
                        c_low_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* HiLoSplitter_transform_bwd (HiLoSplitterObject * self, PyObject * args)
{
  HiLoSplitter* object = self->object;
  ASSERT(object != NULL, "HiLoSplitter NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("(ii)i", object->size(),
                                                       object->size_lowpass(),
                                                       object->size());

  LOG("parsing arguments");
  PyArrayObject * high_in;
  PyArrayObject * low_in;
  PyArrayObject * full_out;

  if (not PyArg_ParseTuple(args, "(O!O!)O!",
                           &PyArray_Type, &high_in,
                           &PyArray_Type, &low_in,
                           &PyArray_Type, &full_out)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS(high_in, object->size());
  ASSERT_REALS(low_in, object->size_lowpass());
  ASSERT_REALS(full_out, object->size());

  LOG("fusing high + low --> full");
  Vector<float> c_high_in = c_Vector_float(high_in);
  Vector<float> c_low_in = c_Vector_float(low_in);
  Vector<float> c_full_out = c_Vector_float(full_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_bwd(c_high_in,
                        c_low_in,
                        c_full_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( shepard scale )-------------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Shepard)

static int Shepard_init (
    ShepardObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size;
  float min_freq_hz = DEFAULT_MIN_FREQ;
  float max_freq_hz = DEFAULT_MAX_FREQ;
  ASSERT_CONSTRUCTOR(Shepard,
          PyArg_ParseTuple(args, "i|ff",
                           &size,
                           &min_freq_hz,
                           &max_freq_hz));
  ASSERT((0 < min_freq_hz),
         "bad frequency range: " << min_freq_hz << " -- " << max_freq_hz);

  LOG("building Shepard(" << size << ", "
                          << min_freq_hz << ", "
                          << max_freq_hz << ")");
  ASSERT(self->object == NULL, "Shepard not NULL on init");
  self->object = new Shepard(size, min_freq_hz, max_freq_hz);

  return 0;
}

//----( specific methods )----

INT_GETTER(Shepard, size_in)
INT_GETTER(Shepard, size_out)

PyObject* Shepard_transform_fwd (ShepardObject * self, PyObject * args)
{
  Shepard* object = self->object;
  ASSERT(object != NULL, "Shepard NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size_in(),
                                                    object->size_out());

  LOG1("parsing arguments");
  PyArrayObject * pitch_in;
  PyArrayObject * tone_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &pitch_in,
                           &PyArray_Type, &tone_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(pitch_in, object->size_in());
  ASSERT_REALS(tone_out, object->size_out());

  LOG1("converting pitch --> tone");
  Vector<float> c_pitch_in = c_Vector_float(pitch_in);
  Vector<float> c_tone_out = c_Vector_float(tone_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_fwd(c_pitch_in,
                        c_tone_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Shepard_transform_bwd (ShepardObject * self, PyObject * args)
{
  Shepard* object = self->object;
  ASSERT(object != NULL, "Shepard NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size_out(),
                                                    object->size_in());

  LOG("parsing arguments");
  PyArrayObject * tone_in;
  PyArrayObject * pitch_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &tone_in,
                           &PyArray_Type, &pitch_out)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS(tone_in, object->size_out());
  ASSERT_REALS(pitch_out, object->size_in());

  LOG("converting tone --> pitch");
  Vector<float> c_tone_in = c_Vector_float(tone_in);
  Vector<float> c_pitch_out = c_Vector_float(pitch_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_bwd(c_tone_in,
                        c_pitch_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( loudness transform )--------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Loudness)

static int Loudness_init (
    LoudnessObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size;
  float frame_rate;
  ASSERT_CONSTRUCTOR(Loudness,
          PyArg_ParseTuple(args, "if",
                           &size,
                           &frame_rate));
  ASSERT(0 < frame_rate,
         "Loudness has nonpositive frame_rate: " << frame_rate);

  LOG("building Loudness : " << size << ", frame_rate " << frame_rate);
  ASSERT(self->object == NULL, "Loudness not NULL on init");
  self->object = new Loudness(size, frame_rate);

  return 0;
}

//----( specific methods )----

INT_GETTER(Loudness, size)
FLOAT_GETTER(Loudness, frame_rate)
FLOAT_GETTER(Loudness, time_scale)
FLOAT_GETTER(Loudness, ss_factor)

PyObject* Loudness_transform_fwd (LoudnessObject * self, PyObject * args)
{
  Loudness* object = self->object;
  ASSERT(object != NULL, "Loudness NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size(),
                                                    object->size());

  LOG1("parsing arguments");
  PyArrayObject * energy_in;
  PyArrayObject * loudness_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &energy_in,
                           &PyArray_Type, &loudness_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(energy_in, object->size());
  ASSERT_REALS(loudness_out, object->size());

  LOG1("converting energy --> loudness");
  Vector<float> c_energy_in = c_Vector_float(energy_in);
  Vector<float> c_loudness_out = c_Vector_float(loudness_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_fwd(c_energy_in,
                        c_loudness_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Loudness_transform_bwd (LoudnessObject * self, PyObject * args)
{
  Loudness* object = self->object;
  ASSERT(object != NULL, "Loudness NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size(),
                                                    object->size());

  LOG("parsing arguments");
  PyArrayObject * loudness_in;
  PyArrayObject * energy_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &loudness_in,
                           &PyArray_Type, &energy_out)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS(loudness_in, object->size());
  ASSERT_REALS(energy_out, object->size());

  LOG("converting loudness --> energy");
  Vector<float> c_loudness_in = c_Vector_float(loudness_in);
  Vector<float> c_energy_out = c_Vector_float(energy_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_bwd(c_loudness_in,
                        c_energy_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( sharpener )-----------------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Sharpener)

static int Sharpener_init (
    SharpenerObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size;
  ASSERT_CONSTRUCTOR(Sharpener,
          PyArg_ParseTuple(args, "i",
                           &size));

  LOG("building Sharpener : " << size);
  ASSERT(self->object == NULL, "Sharpener not NULL on init");
  self->object = new Sharpener(size);

  return 0;
}

//----( specific methods )----

INT_GETTER(Sharpener, size)

PyObject* Sharpener_transform (SharpenerObject * self, PyObject * args)
{
  Sharpener* object = self->object;
  ASSERT(object != NULL, "Sharpener NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size(),
                                                    object->size());

  LOG1("parsing arguments");
  PyArrayObject * freq_in;
  PyArrayObject * freq_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &freq_in,
                           &PyArray_Type, &freq_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(freq_in, object->size());
  ASSERT_REALS(freq_out, object->size());

  LOG1("sharpening spectrum");
  Vector<float> c_freq_in = c_Vector_float(freq_in);
  Vector<float> c_freq_out = c_Vector_float(freq_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform(c_freq_in,
                    c_freq_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( octave shift )--------------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(OctaveLower)

static int OctaveLower_init (
    OctaveLowerObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size;
  ASSERT_CONSTRUCTOR(OctaveLower,
          PyArg_ParseTuple(args, "i",
                           &size));

  LOG("building OctaveLower : " << size);
  ASSERT(self->object == NULL, "OctaveLower not NULL on init");
  self->object = new OctaveLower(size);

  return 0;
}

//----( specific methods )----

INT_GETTER(OctaveLower, size);

PyObject* OctaveLower_transform (OctaveLowerObject * self, PyObject * args)
{
  OctaveLower* object = self->object;
  ASSERT(object != NULL, "OctaveLower NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size(),
                                                    object->size());

  LOG1("parsing arguments");
  PyArrayObject * sound_in;
  PyArrayObject * sound_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &sound_in,
                           &PyArray_Type, &sound_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_COMPLEXES(sound_in, object->size());
  ASSERT_COMPLEXES(sound_out, object->size());

  LOG1("lowering pitch by one octave");
  Vector<complex> c_sound_in = c_Vector_complex(sound_in);
  Vector<complex> c_sound_out = c_Vector_complex(sound_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform(c_sound_in,
                    c_sound_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( pitch shift )---------------------------------------------------------
/* OLD
//----( new, init, dealloc )----

BOILERPLATE_NEW(PitchShift)

static int PitchShift_init (
    PitchShiftObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size_exponent;
  int factor_exponent;
  float halftone_shift;
  ASSERT_CONSTRUCTOR(PitchShift,
          PyArg_ParseTuple(args, "iif",
                           &size_exponent,
                           &factor_exponent,
                           &halftone_shift));

  LOG("building PitchShift : 2^(" << size_exponent << " += "
                                  << factor_exponent << ")");
  ASSERT(self->object == NULL, "PitchShift not NULL on init");
  self->object = new PitchShift(size_exponent,
                                factor_exponent,
                                halftone_shift);

  return 0;
}

//----( specific methods )----

INT_GETTER(PitchShift, size);
FLOAT_GETTER(PitchShift, factor);

PyObject* PitchShift_transform (PitchShiftObject * self, PyObject * args)
{
  PitchShift* object = self->object;
  ASSERT(object != NULL, "PitchShift NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size(),
                                                    object->size());

  LOG1("parsing arguments");
  PyArrayObject * sound_in;
  PyArrayObject * sound_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &sound_in,
                           &PyArray_Type, &sound_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_COMPLEXES(sound_in, object->size());
  ASSERT_COMPLEXES(sound_out, object->size());

  LOG1("shifting pitch");
  Vector<complex> c_sound_in = c_Vector_complex(sound_in);
  Vector<complex> c_sound_out = c_Vector_complex(sound_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform(c_sound_in,
                    c_sound_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}
*/

//----( melodigram objects )--------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Melodigram)

static int Melodigram_init (
    MelodigramObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int exponent;
  int num_filters;
  float frame_rate;
  ASSERT_CONSTRUCTOR(Melodigram,
          PyArg_ParseTuple(args, "iif",
                           &exponent,
                           &num_filters,
                           &frame_rate));
  ASSERT((0 < exponent) and (exponent < MAX_EXPONENT),
         "exponent out of range: " << exponent);
  ASSERT(0 < num_filters,
         "num_filters out of range: " << num_filters);
  ASSERT(0 < frame_rate,
         "frame_rate out of range: " << frame_rate);

  size_t size = 1 << exponent;
  LOG("building Melodigram : "
      << size << " x "
      << num_filters << " filters, frame rate "
      << frame_rate << "Hz");
  ASSERT(self->object == NULL, "Melodigram not NULL on init");
  self->object = new Melodigram(exponent, num_filters, frame_rate);

  return 0;
}

//----( specific methods )----

INT_GETTER(Melodigram, size)
INT_GETTER(Melodigram, size_corr)
INT_GETTER(Melodigram, num_filters)
INT_GETTER(Melodigram, size_out)

PyObject* Melodigram_transform_fwd (MelodigramObject * self, PyObject * args)
{
  Melodigram* object = self->object;
  ASSERT(object != NULL, "Melodigram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size(),
                                                      object->size_out());

  LOG("parsing arguments");
  PyArrayObject * pitch_in;
  PyArrayObject * corr_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &pitch_in,
                           &PyArray_Type, &corr_out)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS(pitch_in, object->size());
  ASSERT_REALS(corr_out, object->size_out());

  LOG("computing melodigram transform");
  Vector<float> c_pitch_in = c_Vector_float(pitch_in);
  Vector<float> c_corr_out = c_Vector_float(corr_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_fwd(c_pitch_in,
                        c_corr_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Melodigram_transform_bwd (MelodigramObject * self, PyObject * args)
{
  Melodigram* object = self->object;
  ASSERT(object != NULL, "Melodigram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("(ii)i", object->size(),
                                                         object->size_out(),
                                                         object->size());

  LOG("parsing arguments");
  PyArrayObject * prev_pitch_in;
  PyArrayObject * corr_in;
  PyArrayObject * pitch_out;

  if (not PyArg_ParseTuple(args, "(O!O!)O!",
                           &PyArray_Type, &prev_pitch_in,
                           &PyArray_Type, &corr_in,
                           &PyArray_Type, &pitch_out)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS(prev_pitch_in, object->size());
  ASSERT_REALS(corr_in, object->size_out());
  ASSERT_REALS(pitch_out, object->size());

  LOG("inverting melodigram transform");
  Vector<float> c_prev_pitch_in = c_Vector_float(prev_pitch_in);
  Vector<float> c_corr_in = c_Vector_float(corr_in);
  Vector<float> c_pitch_out = c_Vector_float(pitch_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_bwd(c_prev_pitch_in,
                        c_corr_in,
                        c_pitch_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( rhythmgram objects )--------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Rhythmgram)

static int Rhythmgram_init (
    RhythmgramObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size_in;
  int size_factor;
  ASSERT_CONSTRUCTOR(Rhythmgram,
          PyArg_ParseTuple(args, "ii",
                           &size_in,
                           &size_factor));
  ASSERT(0 < size_factor,
         "size_factor out of range: " << size_factor);
  ASSERT((0 < size_in) and (size_in * size_factor < MAX_SIZE),
         "size_in out of range: " << size_in);

  LOG("building Rhythmgram : " << size_in << " x " << size_factor);
  ASSERT(self->object == NULL, "Rhythmgram not NULL on init");
  self->object = new Rhythmgram(size_in, size_factor);

  return 0;
}

//----( specific methods )----

INT_GETTER(Rhythmgram, size_in)
INT_GETTER(Rhythmgram, size_factor)
INT_GETTER(Rhythmgram, size_out)

PyObject* Rhythmgram_transform_fwd (RhythmgramObject * self, PyObject * args)
{
  Rhythmgram* object = self->object;
  ASSERT(object != NULL, "Rhythmgram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size_in(),
                                                    object->size_out());

  LOG("parsing arguments");
  PyArrayObject * value_in;
  PyArrayObject * tempo_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &value_in,
                           &PyArray_Type, &tempo_out)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS(value_in, object->size_in());
  ASSERT_REALS(tempo_out, object->size_out());

  LOG("computing rhythmgram transform");
  Vector<float> c_value_in = c_Vector_float(value_in);
  Vector<float> c_tempo_out = c_Vector_float(tempo_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_fwd(c_value_in,
                        c_tempo_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Rhythmgram_transform_bwd (RhythmgramObject * self, PyObject * args)
{
  Rhythmgram* object = self->object;
  ASSERT(object != NULL, "Rhythmgram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size_out(),
                                                      object->size_in());

  LOG("parsing arguments");
  PyArrayObject * tempo_in;
  PyArrayObject * value_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &tempo_in,
                           &PyArray_Type, &value_out)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS(tempo_in, object->size_out());
  ASSERT_REALS(value_out, object->size_in());

  LOG("inverting rhythmgram transform");
  Vector<float> c_tempo_in = c_Vector_float(tempo_in);
  Vector<float> c_value_out = c_Vector_float(value_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_bwd(c_tempo_in,
                        c_value_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( correlogram objects )-------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Correlogram)

static int Correlogram_init (
    CorrelogramObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size;
  float decay_factor;
  ASSERT_CONSTRUCTOR(Correlogram,
          PyArg_ParseTuple(args, "if",
                           &size,
                           &decay_factor));
  ASSERT((0 < size) and (size < (1<<MAX_EXPONENT)),
         "size out of range: " << size);
  ASSERT((0 < decay_factor) and (decay_factor < 1),
         "decay_factor out of range: " << decay_factor);

  LOG("building Correlogram : " << size << ", decay factor "
                                << decay_factor);
  ASSERT(self->object == NULL, "Correlogram not NULL on init");
  self->object = new Correlogram(size, decay_factor);

  return 0;
}

//----( specific methods )----

INT_GETTER(Correlogram, size_in)
INT_GETTER(Correlogram, size_out)
FLOAT_GETTER(Correlogram, decay_factor)

PyObject* Correlogram_transform_fwd (CorrelogramObject * self, PyObject * args)
{
  Correlogram* object = self->object;
  ASSERT(object != NULL, "Correlogram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size_in(),
                                                    object->size_out());

  LOG("parsing arguments");
  PyArrayObject * freq_in;
  PyArrayObject * corr_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &freq_in,
                           &PyArray_Type, &corr_out)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS(freq_in, object->size_in());
  ASSERT_REALS(corr_out, object->size_out());

  LOG("converting frequency --> tone");
  Vector<float> c_freq_in = c_Vector_float(freq_in);
  Vector<float> c_corr_out = c_Vector_float(corr_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_fwd(c_freq_in,
                        c_corr_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Correlogram_transform_bwd (CorrelogramObject * self, PyObject * args)
{
  Correlogram* object = self->object;
  ASSERT(object != NULL, "Correlogram NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("(ii)i", object->size_in(),
                                                       object->size_out(),
                                                       object->size_in());

  LOG("parsing arguments");
  PyArrayObject * prev_freq_in;
  PyArrayObject * corr_in;
  PyArrayObject * freq_out;

  if (not PyArg_ParseTuple(args, "(O!O!)O!",
                           &PyArray_Type, &prev_freq_in,
                           &PyArray_Type, &corr_in,
                           &PyArray_Type, &freq_out)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS(prev_freq_in, object->size_in());
  ASSERT_REALS(corr_in, object->size_out());
  ASSERT_REALS(freq_out, object->size_in());

  LOG("converting tone --> frequency");
  Vector<float> c_prev_freq_in = c_Vector_float(prev_freq_in);
  Vector<float> c_corr_in = c_Vector_float(corr_in);
  Vector<float> c_freq_out = c_Vector_float(freq_out);
  Py_BEGIN_ALLOW_THREADS
  object->transform_bwd(c_prev_freq_in,
                        c_corr_in,
                        c_freq_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( logarithmic history objects )-----------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(History)

static int History_init (
    HistoryObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size;
  int length;
  int density;
  ASSERT_CONSTRUCTOR(History,
          PyArg_ParseTuple(args, "iii",
                           &size,
                           &length,
                           &density));
  ASSERT(size >= 1, "size out of range: " << size);
  ASSERT(length >= 2, "length out of range: " << length);
  ASSERT(density >= 1, "density out of range: " << density);

  LOG("building History(" << size << ", " << length << ", " << density << ')');
  ASSERT(self->object == NULL, "History not NULL on init");
  self->object = new History(size, length, density);

  return 0;
}

//----( specific methods )----

INT_GETTER(History, size);
INT_GETTER(History, length);
INT_GETTER(History, density);
INT_GETTER(History, size_in);
INT_GETTER(History, size_out);
BOOL_GETTER(History, full);

PyObject* History_shape (HistoryObject * self)
{
  History* object = self->object;
  ASSERT(object != NULL, "History NULL");

  PyObject * result = Py_BuildValue("(ii)", object->length(),
                                            object->size());
  Py_INCREF(result);
  return result;
}

PyObject* History_add (HistoryObject * self, PyObject * args)
{
  History * object = self->object;
  ASSERT(object != NULL, "History NULL");

  LOG1("parsing arguments");
  PyArrayObject * frame_in;

  if (not PyArg_ParseTuple(args, "O!",
                           &PyArray_Type, &frame_in)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(frame_in, object->size());

  LOG1("adding frame to history");
  Vector<float> c_frame_in = c_Vector_float(frame_in);
  Py_BEGIN_ALLOW_THREADS
  object->add(c_frame_in);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* History_get (HistoryObject * self, PyObject * args)
{
  History * object = self->object;
  ASSERT(object != NULL, "History NULL");

  LOG1("parsing arguments");
  PyArrayObject * history_out;

  if (not PyArg_ParseTuple(args, "O!",
                           &PyArray_Type, &history_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS2(history_out, object->length(), object->size());

  LOG1("getting history");
  Vector<float> c_history_out = c_Vector_float(history_out);
  Py_BEGIN_ALLOW_THREADS
  object->get(c_history_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* History_transform (HistoryObject * self, PyObject * args)
{
  History * object = self->object;
  ASSERT(object != NULL, "History NULL");

  LOG1("parsing arguments");
  PyArrayObject * frame_in;
  PyArrayObject * history_out;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &frame_in,
                           &PyArray_Type, &history_out)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(frame_in, object->size());
  ASSERT_REALS2(history_out, object->length(), object->size());

  LOG1("running history transform");
  Vector<float> c_frame_in = c_Vector_float(frame_in);
  Vector<float> c_history_out = c_Vector_float(history_out);
  Py_BEGIN_ALLOW_THREADS
  object->add(c_frame_in);
  object->get(c_history_out);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( 1d spline objects )---------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Spline)

static int Spline_init (
    SplineObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size_in;
  int size_out;
  PyArrayObject * fun = NULL;
  ASSERT_CONSTRUCTOR(Spline,
          PyArg_ParseTuple(args, "ii|O!",
                           &size_in,
                           &size_out,
                           &PyArray_Type, &fun));
  ASSERT(1 <= size_in, "size_out out of range: " << size_in);
  ASSERT(1 <= size_out, "size_in out of range: " << size_out);
  if (fun != NULL) {
    ASSERT_REALS(fun, static_cast<size_t>(size_in));
  }

  LOG("building Spline : " << size_in << " --> " << size_out);
  ASSERT(self->object == NULL, "Spline not NULL on init");
  if (fun == NULL) {
    self->object = new Spline(size_in,
                                size_out);
  } else {
    Vector<float> c_fun = c_Vector_float(fun);
    self->object = new Spline(size_in,
                              size_out,
                              c_fun);
  }

  return 0;
}

//----( specific methods )----

INT_GETTER(Spline, size_in)
INT_GETTER(Spline, size_out)

PyObject* Spline_swap (SplineObject * self, PyObject * args)
{
  Spline * object = self->object;
  ASSERT(object != NULL, "Spline NULL");

  LOG1("parsing arguments");
  SplineObject * other;
  if (not PyArg_ParseTuple(args, "O!",
                           &SplineType, &other)) return NULL;

  LOG1("swapping data");
  self->object = other->object;
  other->object = object;

  LOG("swapping Spline : " << self->object->size_in() << " --> "
                           << self->object->size_out());

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Spline_transform_fwd (SplineObject * self, PyObject * args)
{
  Spline * object = self->object;
  ASSERT(object != NULL, "Spline NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size_in(),
                                                    object->size_out());

  LOG1("parsing arguments");
  PyArrayObject * e_dom;
  PyArrayObject * e_rng;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &e_dom,
                           &PyArray_Type, &e_rng)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(e_dom, object->size_in());
  ASSERT_REALS(e_rng, object->size_out());

  LOG1("running forward 1D object");
  Vector<float> c_e_dom = c_Vector_float(e_dom);
  Vector<float> c_e_rng = c_Vector_float(e_rng);
  Py_BEGIN_ALLOW_THREADS
  object->transform_fwd(c_e_dom,
                        c_e_rng);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Spline_transform_bwd (SplineObject * self, PyObject * args)
{
  Spline * object = self->object;
  ASSERT(object != NULL, "Spline NULL");

  if (PyArg_Empty(args)) return Py_BuildValue("ii", object->size_out(),
                                                    object->size_in());

  LOG1("parsing arguments");
  PyArrayObject * e_rng;
  PyArrayObject * e_dom;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &e_rng,
                           &PyArray_Type, &e_dom)) return NULL;

  LOG1("checking arrays");
  ASSERT_REALS(e_rng, object->size_out());
  ASSERT_REALS(e_dom, object->size_in());

  LOG1("running backward 1D object");
  Vector<float> c_e_rng = c_Vector_float(e_rng);
  Vector<float> c_e_dom = c_Vector_float(e_dom);
  Py_BEGIN_ALLOW_THREADS
  object->transform_bwd(c_e_rng,
                        c_e_dom);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( 2d spline objects )---------------------------------------------------

//----( new, init, dealloc )----

BOILERPLATE_NEW(Spline2DSeparable)

static int Spline2DSeparable_init (
    Spline2DSeparableObject * self,
    PyObject * args,
    PyObject * kwds)
{
  int size_in1;
  int size_in2;
  int size_out1;
  int size_out2;
  ASSERT_CONSTRUCTOR(Spline2DSeparable,
          PyArg_ParseTuple(args, "iiii",
                           &size_in1,
                           &size_in2,
                           &size_out1,
                           &size_out2));
  ASSERT(size_in1 >= 1, "size_in1 out of range: " << size_in1);
  ASSERT(size_in2 >= 1, "size_in2 out of range: " << size_in2);
  ASSERT(size_out1 >= 2, "size_out1 out of range: " << size_out1);
  ASSERT(size_out2 >= 2, "size_out2 out of range: " << size_out2);

  LOG("building Spline2DSeparable");
  ASSERT(self->object == NULL, "Spline2DSeparable not NULL on init");
  self->object = new Spline2DSeparable(size_in1,size_in2,size_out1,size_out2);

  return 0;
}

//----( specific methods )----

PyObject* Spline2DSeparable_shape (Spline2DSeparableObject * self)
{
  Spline2DSeparable* object = self->object;
  ASSERT(object != NULL, "Spline2DSeparable NULL");

  PyObject * result = Py_BuildValue("(iiii)", object->size_in1(),
                                              object->size_in2(),
                                              object->size_out1(),
                                              object->size_out2());
  Py_INCREF(result);
  return result;
}

PyObject* Spline2DSeparable_transform_fwd (Spline2DSeparableObject * self, PyObject * args)
{
  Spline2DSeparable * object = self->object;
  ASSERT(object != NULL, "Spline2DSeparable NULL");

  LOG("parsing arguments");
  PyArrayObject * e_dom;
  PyArrayObject * e_rng;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &e_dom,
                           &PyArray_Type, &e_rng)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS2(e_dom, object->size_in1(), object->size_in2());
  ASSERT_REALS2(e_rng, object->size_out1(), object->size_out2());

  LOG("running forward 2D object");
  Vector<float> c_e_dom = c_Vector_float(e_dom);
  Vector<float> c_e_rng = c_Vector_float(e_rng);
  Py_BEGIN_ALLOW_THREADS
  object->transform_fwd(c_e_dom,
                        c_e_rng);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

PyObject* Spline2DSeparable_transform_bwd (Spline2DSeparableObject * self, PyObject * args)
{
  Spline2DSeparable * object = self->object;
  ASSERT(object != NULL, "Spline2DSeparable NULL");

  LOG("parsing arguments");
  PyArrayObject * e_rng;
  PyArrayObject * e_dom;

  if (not PyArg_ParseTuple(args, "O!O!",
                           &PyArray_Type, &e_rng,
                           &PyArray_Type, &e_dom)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS2(e_rng, object->size_out1(), object->size_out2());
  ASSERT_REALS2(e_dom, object->size_in1(), object->size_in2());

  LOG("running backward 2D object");
  Vector<float> c_e_rng = c_Vector_float(e_rng);
  Vector<float> c_e_dom = c_Vector_float(e_dom);
  Py_BEGIN_ALLOW_THREADS
  object->transform_bwd(c_e_rng,
                        c_e_dom);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

//----( misc functions )------------------------------------------------------

#define MAX_DIMS 4

PyObject * new_Vector_float (PyObject * self, PyObject * args)
{
  int sizes[MAX_DIMS+1] = {0,0,0,0,0};
  if (not PyArg_ParseTuple(args, "i|iii",
                           sizes,
                           sizes + 1,
                           sizes + 2,
                           sizes + 3)) return NULL;

  size_t nd = 0;
  int size = 1;
  npy_intp dims[MAX_DIMS];
  while (sizes[nd]) {
    size *= sizes[nd];
    dims[nd] = sizes[nd];
    ++nd;
  }
  float * data = malloc_float(size);
  zero_float(data, size);
  PyObject * result = PyArray_SimpleNewFromData(nd, dims, PyArray_FLOAT, data);

  // WARNING HACK data is never deallocated

  Py_INCREF(result);
  return result;
}

PyObject * new_Vector_complex (PyObject * self, PyObject * args)
{
  int sizes[MAX_DIMS+1] = {0,0,0,0,0};
  if (not PyArg_ParseTuple(args, "i|iii",
                           sizes,
                           sizes + 1,
                           sizes + 2,
                           sizes + 3)) return NULL;

  size_t nd = 0;
  int size = 1;
  npy_intp dims[MAX_DIMS];
  while (sizes[nd]) {
    size *= sizes[nd];
    dims[nd] = sizes[nd];
    ++nd;
  }
  complex * data = malloc_complex(size);
  zero_complex(data, size);
  PyObject * result = PyArray_SimpleNewFromData(nd, dims, PyArray_CFLOAT, data);

  // WARNING HACK data is never deallocated

  Py_INCREF(result);
  return result;
}

PyObject * hdr_real_color (PyObject * self, PyObject * args)
{
  LOG("parsing arguments");
  int I,J;
  PyArrayObject * r;
  PyArrayObject * g;
  PyArrayObject * b;

  if (not PyArg_ParseTuple(args, "iiO!O!O!",
                           &I, &J,
                           &PyArray_Type, &r,
                           &PyArray_Type, &g,
                           &PyArray_Type, &b)) return NULL;

  LOG("checking arrays");
  ASSERT_REALS2(r, I,J);
  ASSERT_REALS2(g, I,J);
  ASSERT_REALS2(b, I,J);

  LOG("running forward 2D object");
  Vector<float> c_r = c_Vector_float(r);
  Vector<float> c_g = c_Vector_float(g);
  Vector<float> c_b = c_Vector_float(b);
  Py_BEGIN_ALLOW_THREADS
  Image::hdr_real_color(I,J, c_r,c_g,c_b);
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  return Py_None;
}

