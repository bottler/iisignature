#ifdef __MINGW32__
  #define _hypot hypot
  #include <cmath>
#endif

#if defined(_WINDOWS) && defined(_DEBUG)
//Not having a debug build of numpy shouldn't stop
//me being able to debug the other parts of the library
 #define IISIGNATURE_NO_NUMPY
#endif

#include<utility>
#include<vector>
#include<utility>
#include<iostream>
#include<memory>
#include<limits>
#include<sstream>
#include<string>
#include<Python.h>
#ifndef IISIGNATURE_NO_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<numpy/arrayobject.h>
#endif

#include "calcSignature.hpp"
#include "logSigLength.hpp"
#include "logsig.hpp"
#include "rotationalInvariants.hpp"

using std::vector;

#if PY_MAJOR_VERSION <3 && PY_MINOR_VERSION<7
  #define NO_CAPSULES
#endif

// for python 3.5+, we can load the new way https://www.python.org/dev/peps/pep-0489/

/* 
template<typename T> int numpyTypenum;
template<> int numpyTypenum<float>  = NPY_FLOAT32;
template<> int numpyTypenum<double>  = NPY_FLOAT64;

template<int> class numpyTypeImpl;
template<> class numpyTypeImpl<NPY_FLOAT32>{public: typedef float Type;};
template<> class numpyTypeImpl<NPY_FLOAT64>{public: typedef double Type;};
template<int i> using numpyNumToType = typename numpyTypeImpl<i>::Type;
*/

#define ERR(X) {PyErr_SetString(PyExc_RuntimeError,X); return nullptr;}
#define ERRb(X) {PyErr_SetString(PyExc_RuntimeError,X); return false;}
#define ERRr(X) {PyErr_SetString(PyExc_RuntimeError,X); return;}

//INTERRUPTS - the following seems to be fine even on windows
#include <signal.h>
volatile bool g_signals_setup=false; 
volatile bool g_signal_given=false; 
void catcher(int){
  g_signal_given = true;
  PyErr_SetInterrupt();
}

void setup_signals(){
  g_signal_given = false;
  if(g_signals_setup)
    return;
  signal(SIGINT,&catcher);
  g_signals_setup = true;
}
bool interrupt_wanted(){return g_signal_given;}
void interrupt(){if(g_signal_given) throw std::runtime_error("INTERRUPTION");}

//The following 3 functions just do t, which returns true if it succeeded,
// returning true if t succeeded.
//When in one of the python functions (defined in this file), 
//calling a function which might call interrupt needs catching like this.
template<typename T>
bool do_interruptible(T&& t) {
  bool ok = false;
  setup_signals();
  std::string exceptionMessage;
  try {
    ok = t();
  }
  catch (std::exception& e) {
    exceptionMessage = e.what();
  }
  if (PyErr_CheckSignals())
    return false;
  if (!exceptionMessage.empty())
    ERRb(exceptionMessage.c_str());
  return ok;
}
template<typename T>
bool do_dummy(T&& t) {//useful for timing experiments, 
  return t();
}
//Like do_interruptible but also releases the GIL
template<typename T>
bool do_interruptible_releasing_lock(T&& t) {
  bool ok = false;
  setup_signals();
  std::string exceptionMessage;
  Py_BEGIN_ALLOW_THREADS
  try {
    ok = t();
  }
  catch (std::exception& e) {
    exceptionMessage = e.what();
  }
  Py_END_ALLOW_THREADS
  if (PyErr_CheckSignals())
    return false;
  if (!exceptionMessage.empty())
    ERRb(exceptionMessage.c_str());
  return ok;
}

//end of Interrupts stuff

//This class is constructed from a PyObject* and decreases
//its reference count on destruction.
//It is basically an "owner" of a reference.
class RefHolder{
  PyObject* m_p;
public:
  RefHolder(PyObject* p):m_p(p){};
  RefHolder(const RefHolder&)=delete;
  RefHolder operator=(const RefHolder&) = delete;
  ~RefHolder(){Py_DECREF(m_p);}
};

//This is a version which can change what it owns (or whether it owns anything).
class ReleasableRefHolder {
  PyObject* m_p;
public:
  ReleasableRefHolder(PyObject* p) :m_p(p) {};
  ReleasableRefHolder() : m_p(nullptr) {};
  void releaseAndSet(PyObject* p) { m_p = p; }
  void release() { m_p = nullptr; }
  ReleasableRefHolder(const ReleasableRefHolder&) = delete;
  ReleasableRefHolder operator=(const ReleasableRefHolder&) = delete;
  ~ReleasableRefHolder() { Py_XDECREF(m_p); }
};

#ifndef IISIGNATURE_NO_NUMPY
struct UseFloat{
  constexpr static int typenum=NPY_FLOAT32;
  using T = float;
};

struct UseDouble{
  constexpr static int typenum=NPY_FLOAT64;
  using T = double;
};

//Returns a new numpy array with the type typenum,
//with ndims dimensions. The first (ndims-1) dimensions are the same
//as those of arr, and the last dimension is newLastDim.
//We save an allocation by reusing the shape member of arr,
//so ndims must not exceed the number of dimensions of arr.
PyObject* simpleNew_ownLastDim(int ndims, PyArrayObject* arr, size_t newLastDim, int typenum) {
  npy_intp* shape = PyArray_DIMS(arr);
  npy_intp orig = shape[ndims - 1];
  shape[ndims - 1] = (npy_intp) newLastDim;
  PyObject* o = PyArray_SimpleNew(ndims, shape, typenum);
  shape[ndims - 1] = orig;
  return o;
}
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
static PyObject *
version(PyObject *self, PyObject *args){
  //consider returning build time?
  return PyUnicode_FromString(TOSTRING(VERSION));
}

static PyObject *
siglength(PyObject *self, PyObject *args){
  int d=0, m=0;
  if (!PyArg_ParseTuple(args, "ii", &d, &m))
    return nullptr;
  if(m<1) ERR("level must be positive");
  if(d<1) ERR("dimension must be positive");
  long ans = calcSigTotalLength(d,m);
  //todo - cope with overrun here
#if PY_MAJOR_VERSION >= 3
  return PyLong_FromLong(ans);
#else
  return PyInt_FromLong(ans);
#endif
}


static PyObject *
logsiglength(PyObject *self, PyObject *args){
  int d=0, m=0;
  if (!PyArg_ParseTuple(args, "ii", &d, &m))
    return nullptr;
  if(m<1) ERR("level must be positive");
  if(d<1) ERR("dimension must be positive");
  LogSigLength::Int ans = d==1 ? 1 : m==1 ? d : LogSigLength::countNecklacesUptoLengthM(d,m);
  if(ans>std::numeric_limits<long>::max()) ERR("overflow");
  //todo - cope with overrun here
#if PY_MAJOR_VERSION >= 3
  return PyLong_FromLong((long)ans);
#else
  return PyInt_FromLong((long)ans);
#endif
}

#ifndef IISIGNATURE_NO_NUMPY
//returns true on success
//makes s2 be the signature of the path in data
//data is (lengthOfPath x d)
using CalcSignature::Signature;
static bool calcSignature(Signature& s2, const double* data, int lengthOfPath, int d, int level){
  Signature s1;

  if(lengthOfPath==1){
    s2.sigOfNothing(d,level);
  }

  vector<double> displacement(d);
  for(int i=1; i<lengthOfPath; ++i){
    for(int j=0;j<d; ++j)
      displacement[j]=data[i*d+j]-data[(i-1)*d+j];
    s1.sigOfSegment(d,level,&displacement[0]);
    if (interrupt_wanted())
      return false;
    if(i==1)
      s2.swap(s1);
    else {
      s2.concatenateWith(d, level, s1);
      if (interrupt_wanted())
        return false;
    }
  }
  return true;
}

static PyObject *
sig(PyObject *self, PyObject *args) {
  PyObject* a1;
  int level = 0;
  int format = 0;
  if (!PyArg_ParseTuple(args, "Oi|i", &a1, &level, &format))
    return nullptr;
  if (level < 1) ERR("level must be positive");
  //could have a shortcut here if a1 is a contiguous array of float32
  PyObject* aa = PyArray_ContiguousFromAny(a1, NPY_FLOAT64, 0, 0);
  if (!aa) ERR("data must be (convertable to) a numpy array");
  RefHolder a_(aa);
  PyArrayObject* a = (PyArrayObject*)aa;
  int ndims = PyArray_NDIM(a);
  if (ndims < 2) ERR("data must be 2d");
  const int lengthOfPath = (int)PyArray_DIM(a, ndims - 2);
  const int d = (int)PyArray_DIM(a, ndims - 1);
  if (lengthOfPath < 1) ERR("Path has no length");
  if (d < 1) ERR("Path must have positive dimension");
  int nPaths = 1;
  for (int i = 0; i + 2 < ndims; ++i) {
    npy_intp x = PyArray_DIM(a, i);
    nPaths *= (int)x;
  }
  size_t eachInputSize = (size_t)(lengthOfPath * d);
  size_t eachOutputSize = (size_t)calcSigTotalLength(d, level);
  PyObject* o = nullptr;
  using OutT = UseFloat;
  OutT::T* out_data = nullptr;
  if (format == 0) {
    o = simpleNew_ownLastDim(ndims - 1, a, eachOutputSize, OutT::typenum);
    if (!o)
      return nullptr;
    out_data = static_cast<OutT::T*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o)));
  }
  else if (format != 1) 
    ERR("Invalid format requested");
  ReleasableRefHolder o_(o);
  double* in_data = (double*)PyArray_DATA(a);

  Signature s;
  bool ok = do_interruptible([&]{
    for (int path = 0; path < nPaths; ++path) {
      if (!calcSignature(s, in_data + path*eachInputSize, lengthOfPath, d, level))
        return false;
      if (format == 0)
        s.writeOut(out_data + path * eachOutputSize);
    }
    return true;
  });
  if (!ok)
    return nullptr;

  if (format == 1){
    o = PyTuple_New(level);
    for (int m = 0; m < level; ++m) {
      npy_intp dims[] = { (npy_intp)calcSigLevelLength(d,m+1) };
      PyObject* p = PyArray_SimpleNew(1, dims, OutT::typenum);
      if (!p)
        return nullptr;
      auto ptr = static_cast<OutT::T*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(p)));
      for (auto f : s.m_data[m])
        *(ptr++) = (OutT::T)f;
      PyTuple_SET_ITEM(o, m, p);
    }
  }
  o_.release();
  return o;
}

static PyObject *
sigMultCount(PyObject *self, PyObject *args) {
  PyObject* data;
  int level = 0;
  int format;
  if (!PyArg_ParseTuple(args, "Oi|i", &data, &level, &format))
    return nullptr;
  if (level<1) ERR("level must be positive");
  PyObject* aa = PyArray_ContiguousFromAny(data, NPY_FLOAT64, 0, 0);
  if (!aa) ERR("data must be (convertable to) a numpy array");
  RefHolder a_(aa);
  PyArrayObject* a = (PyArrayObject*)aa;
  if (PyArray_NDIM(a) != 2) ERR("data must be 2d");
  const int lengthOfPath = (int)PyArray_DIM(a, 0);
  const int d = (int)PyArray_DIM(a, 1);
  if (lengthOfPath<1) ERR("Path has no length");
  if (d<1) ERR("Path must have positive dimension");

  double out = CalcSignature::Signature::concatenateWithMultCount(d,level)*(lengthOfPath-2);
  out += CalcSignature::Signature::sigOfSegmentMultCount(d, level)*(lengthOfPath - 1);
  return PyLong_FromDouble(out);
}


//Take a Numpy array which is already ensured contiguous and
//either NPY_FLOAT32 or NPY_FLOAT64
//and read from it as doubles
class ReadArrayAsDoubles{
  vector<double> m_store;
  double* m_ptr = nullptr;
public:
  //returns true on bad_alloc
  bool read(PyArrayObject* a, size_t size){
    if(PyArray_TYPE(a)==NPY_FLOAT64){
      m_ptr = static_cast<double*>(PyArray_DATA(a));
      return false;
    }
    try{
      m_store.resize(size);
    }catch(std::bad_alloc&){
      return true;
    }
    auto source = static_cast<float*>(PyArray_DATA(a));
    for(size_t i=0; i<size; ++i)
      m_store[i]=source[i];
    m_ptr = m_store.data();
    return false;
  }
  const double* ptr() const {return m_ptr;}
};

static PyObject *
sigBackwards(PyObject *self, PyObject *args){
  PyObject* a1;
  PyObject* a2;
  int level=0;
  if (!PyArg_ParseTuple(args, "OOi", &a2, &a1, &level))
    return nullptr;
  if(level<1) ERR("level must be positive");
  PyObject* aa = PyArray_ContiguousFromAny(a1, NPY_FLOAT64, 0, 0);
  if (!aa) ERR("path must be (convertable to) a numpy array");
  RefHolder a_(aa);
  PyArrayObject* a = (PyArrayObject*)aa;
  PyObject* bb = PyArray_ContiguousFromAny(a2, NPY_FLOAT64, 0, 0);
  if (!bb) ERR("derivs must be (convertable to) a numpy array");
  RefHolder b_(bb);
  PyArrayObject* b = (PyArrayObject*)bb;

  int ndims = PyArray_NDIM(a);
  if (ndims < 2) ERR("path must be 2d");
  const int lengthOfPath = (int)PyArray_DIM(a, ndims - 2);
  const int d = (int)PyArray_DIM(a, ndims - 1);
  if (lengthOfPath < 1) ERR("Path has no length");
  if (d < 1) ERR("Path must have positive dimension");
  size_t sigLength = calcSigTotalLength(d, level);
  if (PyArray_NDIM(b) != ndims-1) ERR("derivs has the wrong number of dimensions");
  if (sigLength != (size_t)PyArray_DIM(b, ndims-2))
    ERR(("derivs should have length " + std::to_string(sigLength) +
      " but it has length " + std::to_string(PyArray_DIM(b, ndims-2))).c_str());
  int nPaths = 1;
  for (int i = 0; i + 2 < ndims; ++i) {
    npy_intp x = PyArray_DIM(a, i);
    if (PyArray_DIM(b, i) != x)
      ERR("mismatched dimensions between paths and derivatives");
    nPaths *= (int)x;
  }
  size_t eachInputSize = (size_t)(lengthOfPath * d);
  size_t eachOutputSize = sigLength;

  ReadArrayAsDoubles input, derivs;
  if(input.read(a,lengthOfPath*d) || derivs.read(b,sigLength))
    ERR("Out of memory");
  PyObject* o = PyArray_ZEROS(ndims, PyArray_DIMS(a), NPY_FLOAT32, 0);
  if(!o)
    return nullptr;
  float* out = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o)));
  for(int path =0; path<nPaths; ++path)
    CalcSignature::sigBackwardsRaw(d,level,lengthOfPath,
      input.ptr()+path*eachInputSize,derivs.ptr()+path*eachOutputSize,out+path*eachInputSize);  
  return o;
}

static PyObject *
sigJacobian(PyObject *self, PyObject *args){
  PyObject* a1;
  int level=0;
  if (!PyArg_ParseTuple(args, "Oi", &a1, &level))
    return nullptr;
  if(level<1) ERR("level must be positive");
  PyObject* aa = PyArray_ContiguousFromAny(a1, NPY_FLOAT64, 0, 0);
  if (!aa) ERR("data must be (convertable to) a numpy array");
  RefHolder a_(aa);
  PyArrayObject* a = (PyArrayObject*)aa;
  if(PyArray_NDIM(a)!=2) ERR("data must be 2d");
  const int lengthOfPath = (int)PyArray_DIM(a,0);
  const int d = (int)PyArray_DIM(a,1);
  if(lengthOfPath<1) ERR("Path has no length");
  if(d<1) ERR("Path must have positive dimension");
  size_t sigLength = calcSigTotalLength(d,level);
  ReadArrayAsDoubles input;
  if(input.read(a,lengthOfPath*d))
    ERR("Out of memory");

  npy_intp dims[] = {(npy_intp) lengthOfPath, (npy_intp) d, (npy_intp)sigLength};
  PyObject* o = PyArray_SimpleNew(3,dims,NPY_FLOAT32);
  if(!o)
    return nullptr;
  float* out = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o)));
  TotalDerivativeSignature::sigJacobian(input.ptr(),lengthOfPath,d,level,out);  
  return o;
}

static PyObject *
sigJoin(PyObject *self, PyObject *args){
  PyObject* a1;
  PyObject* a2;
  int level=0;
  double fixedLast = std::numeric_limits<double>::quiet_NaN();
  if (!PyArg_ParseTuple(args, "OOi|d", &a1, &a2, &level, &fixedLast))
    return nullptr;
  if(level<1) ERR("level must be positive");
  PyObject* aa = PyArray_ContiguousFromAny(a1, NPY_FLOAT64, 0, 0);
  if (!aa) ERR("sigs must be (convertable to) a numpy array");
  RefHolder a_(aa);
  PyArrayObject* a = (PyArrayObject*)aa;
  PyObject* bb = PyArray_ContiguousFromAny(a2, NPY_FLOAT64, 0, 0);
  if (!bb) ERR("data must be (convertable to) a numpy array");
  RefHolder b_(bb);
  PyArrayObject* b = (PyArrayObject*)bb;

  int ndims = PyArray_NDIM(a);
  if (ndims < 1) ERR("no signatures provided");
  if (PyArray_NDIM(b) != ndims) ERR("sigs and data must have the same number of dimensions");
  const int d_given = (int)PyArray_DIM(b, ndims-1);
  if (d_given<1) ERR("Path must have positive dimension");
  const int d_out = std::isnan(fixedLast) ? d_given : d_given + 1;
  size_t sigLength = calcSigTotalLength(d_out, level);
  if (sigLength != (size_t)PyArray_DIM(a, ndims-1))
    ERR("signatures have unexpected length");

  int nPaths = 1;
  for (int i = 0; i + 1 < ndims; ++i) {
    npy_intp x = PyArray_DIM(a, i);
    if (PyArray_DIM(b, i) != x)
      ERR("mismatched dimensions between sigs and data");
    nPaths *= (int)x;
  }

  ReadArrayAsDoubles sig, displacement;
  if(sig.read(a,nPaths*sigLength)||displacement.read(b,nPaths*d_given))
    ERR("Out of memory");

  PyObject* o = PyArray_SimpleNew(ndims, PyArray_DIMS(a),NPY_FLOAT32);
  if(!o)
    return nullptr;
  float* out = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o)));
  for(int iPath=0; iPath<nPaths; ++iPath)
    CalcSignature::sigJoin(d_out,level,sig.ptr()+iPath*sigLength,
    displacement.ptr()+iPath*d_given,fixedLast,out+iPath*sigLength);
  return o;
}
 
static PyObject *
  sigJoinBackwards(PyObject* self, PyObject *args){
  PyObject* a1;
  PyObject* a2;
  PyObject* a3;
  int level=0;
  double fixedLast = std::numeric_limits<double>::quiet_NaN();
  if (!PyArg_ParseTuple(args, "OOOi|d", &a3, &a1, &a2, &level, &fixedLast))
    return nullptr;
  if(level<1) ERR("level must be positive");
  PyObject* aa = PyArray_ContiguousFromAny(a1, NPY_FLOAT64, 0, 0);
  if (!aa) ERR("sigs must be (convertable to) a numpy array");
  RefHolder a_(aa);
  PyArrayObject* a = (PyArrayObject*)aa;
  PyObject* bb = PyArray_ContiguousFromAny(a2, NPY_FLOAT64, 0, 0);
  if (!bb) ERR("new data must be (convertable to) a numpy array");
  RefHolder b_(bb);
  PyArrayObject* b = (PyArrayObject*)bb;
  PyObject* cc = PyArray_ContiguousFromAny(a3, NPY_FLOAT64, 0, 0);
  if (!cc) ERR("derivs must be (convertable to) a numpy array");
  RefHolder c_(cc);
  PyArrayObject* c = (PyArrayObject*)cc;

  int ndims = PyArray_NDIM(a);
  if (ndims < 1) ERR("no signatures provided");
  if (PyArray_NDIM(b) != ndims) ERR("sigs and data must have the same number of dimensions");
  if (PyArray_NDIM(c) != ndims) ERR("sigs and derivs must have the same number of dimensions");
  const int d_given = (int)PyArray_DIM(b, ndims - 1);
  if (d_given<1) ERR("Path must have positive dimension");
  const int d_out = std::isnan(fixedLast) ? d_given : d_given + 1;
  size_t sigLength = calcSigTotalLength(d_out, level);
  if (sigLength != (size_t)PyArray_DIM(a, ndims - 1))
    ERR("signatures have unexpected length");

  int nPaths = 1;
  for (int i = 0; i + 1 < ndims; ++i) {
    npy_intp x = PyArray_DIM(a, i);
    if (PyArray_DIM(b, i) != x)
      ERR("mismatched dimensions between sigs and data");
    if (PyArray_DIM(c, i) != x)
      ERR("mismatched dimensions between sigs and derivs");
    nPaths *= (int)x;
  }
  ReadArrayAsDoubles sig, displacement, derivs;
  if(sig.read(a,nPaths*sigLength)||displacement.read(b,nPaths*d_given)
     ||derivs.read(c,nPaths*sigLength))
    ERR("Out of memory");

  PyObject* o1 = PyArray_SimpleNew(ndims, PyArray_DIMS(a),NPY_FLOAT32);
  if(!o1)
    return nullptr;
  PyObject* o2 = PyArray_SimpleNew(ndims, PyArray_DIMS(b), NPY_FLOAT32);
  if (!o2) {
    Py_DECREF(o1);
    return nullptr;
  }
  
  float* out1 = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o1)));
  float* out2 = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o2)));
  double dFixedLast = 0;
  for(int iPath=0; iPath<nPaths; ++iPath)
    CalcSignature::sigJoinBackwards(d_out,level,sig.ptr()+iPath*sigLength,
                                                  displacement.ptr()+iPath*d_given,
                                                  derivs.ptr()+iPath*sigLength,
                                                  fixedLast,
                                                  out1+iPath*sigLength,
                                                  out2+iPath*d_given,
                                                  dFixedLast);
  //PyTuple_Pack doesn't steal references
  if(std::isnan(fixedLast))
    return Py_BuildValue("(NN)", o1, o2);
  else
    return Py_BuildValue("(NNd)", o1, o2, dFixedLast);
}

static PyObject *
sigScale(PyObject *self, PyObject *args){
  PyObject* a1;
  PyObject* a2;
  int level=0;
  if (!PyArg_ParseTuple(args, "OOi", &a1, &a2, &level))
    return nullptr;
  if(level<1) ERR("level must be positive");
  PyObject* aa = PyArray_ContiguousFromAny(a1, NPY_FLOAT64, 0, 0);
  if (!aa) ERR("sigs must be (convertable to) a numpy array");
  RefHolder a_(aa);
  PyArrayObject* a = (PyArrayObject*)aa;
  PyObject* bb = PyArray_ContiguousFromAny(a2, NPY_FLOAT64, 0, 0);
  if (!bb) ERR("scales must be (convertable to) a numpy array");
  RefHolder b_(bb);
  PyArrayObject* b = (PyArrayObject*)bb;

  int ndims = PyArray_NDIM(a);
  if (ndims < 1) ERR("no signatures provided");
  if (PyArray_NDIM(b) != ndims) ERR("sigs and scales must have the same number of dimensions");
  const int d = (int)PyArray_DIM(b, ndims - 1);
  if (d<1) ERR("scales must have positive dimension");
  size_t sigLength = calcSigTotalLength(d, level);
  if (sigLength != (size_t)PyArray_DIM(a, ndims - 1))
    ERR("signatures have unexpected length");

  int nPaths = 1;
  for (int i = 0; i + 1 < ndims; ++i) {
    npy_intp x = PyArray_DIM(a, i);
    if (PyArray_DIM(b, i) != x)
      ERR("mismatched dimensions between sigs and scales");
    nPaths *= (int)x;
  }

  ReadArrayAsDoubles sig, scales;
  if(sig.read(a,nPaths*sigLength)||scales.read(b,nPaths*d))
    ERR("Out of memory");

  PyObject* o = PyArray_SimpleNew(ndims, PyArray_DIMS(a),NPY_FLOAT32);
  if(!o)
    return nullptr;
  float* out = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o)));
  for(int iPath=0; iPath<nPaths; ++iPath)
    CalcSignature::sigScale(d,level,sig.ptr()+iPath*sigLength,
                                          scales.ptr()+iPath*d,out+iPath*sigLength);
  return o;
}

static PyObject *
  sigScaleBackwards(PyObject* self, PyObject *args){
  PyObject* a1;
  PyObject* a2;
  PyObject* a3;
  int level=0;
  if (!PyArg_ParseTuple(args, "OOOi", &a3, &a1, &a2, &level))
    return nullptr;
  if(level<1) ERR("level must be positive");
  PyObject* aa = PyArray_ContiguousFromAny(a1, NPY_FLOAT64, 0, 0);
  if (!aa) ERR("sigs must be (convertable to) a numpy array");
  RefHolder a_(aa);
  PyArrayObject* a = (PyArrayObject*)aa;
  PyObject* bb = PyArray_ContiguousFromAny(a2, NPY_FLOAT64, 0, 0);
  if (!bb) ERR("scales must be (convertable to) a numpy array");
  RefHolder b_(bb);
  PyArrayObject* b = (PyArrayObject*)bb;
  PyObject* cc = PyArray_ContiguousFromAny(a3, NPY_FLOAT64, 0, 0);
  if (!cc) ERR("derivs must be (convertable to) a numpy array");
  RefHolder c_(cc);
  PyArrayObject* c = (PyArrayObject*)cc;

  int ndims = PyArray_NDIM(a);
  if (ndims < 1) ERR("no signatures provided");
  if (PyArray_NDIM(b) != ndims) ERR("sigs and data must have the same number of dimensions");
  if (PyArray_NDIM(c) != ndims) ERR("sigs and derivs must have the same number of dimensions");
  const int d = (int)PyArray_DIM(b, ndims - 1);
  if (d<1) ERR("Path must have positive dimension");
  size_t sigLength = calcSigTotalLength(d, level);
  if (sigLength != (size_t)PyArray_DIM(a, ndims - 1))
    ERR("signatures have unexpected length");

  int nPaths = 1;
  for (int i = 0; i + 1 < ndims; ++i) {
    npy_intp x = PyArray_DIM(a, i);
    if (PyArray_DIM(b, i) != x)
      ERR("mismatched dimensions between sigs and data");
    if (PyArray_DIM(c, i) != x)
      ERR("mismatched dimensions between sigs and derivs");
    nPaths *= (int)x;
  }

  ReadArrayAsDoubles sig, scale, derivs;
  if(sig.read(a,nPaths*sigLength)||scale.read(b,nPaths*d)
     ||derivs.read(c,nPaths*sigLength))
    ERR("Out of memory");

  PyObject* o1 = PyArray_SimpleNew(ndims, PyArray_DIMS(a), NPY_FLOAT32);
  if (!o1)
    return nullptr;
  PyObject* o2 = PyArray_SimpleNew(ndims, PyArray_DIMS(b), NPY_FLOAT32);
  if (!o2) {
    Py_DECREF(o1);
    return nullptr;
  }

  float* out1 = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o1)));
  float* out2 = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o2)));
  for(int iPath=0; iPath<nPaths; ++iPath)
    CalcSignature::sigScaleBackwards(d,level,sig.ptr()+iPath*sigLength,
                                                  scale.ptr()+iPath*d,
                                                  derivs.ptr()+iPath*sigLength,
                                                  out1+iPath*sigLength,
                                                  out2+iPath*d);
  return Py_BuildValue("(NN)", o1, o2);
}
#endif // IISIGNATURE_NO_NUMPY

const char* const logSigFunction_id = "iisignature.LogSigFunction";
LogSigFunction* getLogSigFunction(PyObject* p){
#ifdef NO_CAPSULES
  if(!PyCObject_Check(p))
    ERR("I have received an input which is not from iisignature.prepare()");
  return (LogSigFunction*) PyCObject_AsVoidPtr(p);
#else
  if(!PyCapsule_IsValid(p,logSigFunction_id))
    ERR("I have received an input which is not from iisignature.prepare()");
  return (LogSigFunction*) PyCapsule_GetPointer(p,logSigFunction_id);
#endif
} 

#ifndef NO_CAPSULES
static void killLogSigFunction(PyObject* p){
#else
static void killLogSigFunction(void* v){
  PyObject* p = (PyObject*) v;
#endif
  delete getLogSigFunction(p);
}



//returns true on success
bool getData(){
  if(g_bchLyndon20_dat) //already set
    return true;
  PyObject* get_data = nullptr;
  //PyObject* pkgutil = PyImport_AddModule("pkgutil"); //flaky, returns borrowed
  PyObject* pkgutil = PyImport_ImportModule("pkgutil");
  if(pkgutil){
    RefHolder p_(pkgutil);
    get_data = PyObject_GetAttrString(pkgutil,"get_data");
  }
  if(!get_data){
    PyObject* pkgutil = PyImport_ImportModule("pkg_resources");
    if(pkgutil){
      RefHolder p_(pkgutil);
      get_data = PyObject_GetAttrString(pkgutil,"resource_string");
    }
  }
  if(!get_data)
    ERRb("neither pkgutil nor pkg_resources is working");
  RefHolder get_data_(get_data);
  PyObject* ii = PyUnicode_FromString("iisignature_data");
  RefHolder ii_(ii);
  PyObject* name = PyUnicode_FromString("bchLyndon20.dat");
  RefHolder name_(name);
  PyObject* o = PyObject_CallFunctionObjArgs(get_data,ii,name,NULL);
  if(!o)
    return false;
  if(o==Py_None)
    ERRb("Cannot find iisignature_data module");
#if PY_MAJOR_VERSION <3
  if(!PyString_CheckExact(o))
    ERRb("unexpected type from pkgutil.get_data");
  g_bchLyndon20_dat = PyString_AsString(o);
#else
  if(!PyBytes_CheckExact(o)){
    //std::cerr<<PyBytes_AsString(PyObject_Bytes(o))<<"\n";
    ERRb("unexpected type from pkgutil.get_data");
  }
  g_bchLyndon20_dat = PyBytes_AsString(o);
#endif
  //deliberately leak a reference to o - we'll keep it forever.
  return true;
}

#ifndef IISIGNATURE_NO_NUMPY
//this class just provides access to the functions lstsq, pinv, svd and transpose from numpy
class LeastSquares {
  PyObject* m_transpose;
  PyObject* m_lstsq;
  PyObject* m_pinv;
  PyObject* m_svd;
  PyObject* m_qrScipy = nullptr;
  ReleasableRefHolder m_t_, m_l_, m_p_, m_s_, m_q_; //we can do better than this
public:
  bool m_ok = false;
  LeastSquares(bool wantQR = false) {
    PyObject* numpy = PyImport_AddModule("numpy");
    if (!numpy)
      return;
    PyObject* linalg = PyObject_GetAttrString(numpy, "linalg");
    if (!linalg)
      return;
    RefHolder linalg_(linalg);
    m_transpose = PyObject_GetAttrString(numpy, "transpose");
    if (!m_transpose)
      return;
    m_t_.releaseAndSet(m_transpose);
    m_lstsq = PyObject_GetAttrString(linalg, "lstsq");
    if (!m_lstsq)
      return;
    m_l_.releaseAndSet(m_lstsq);
    m_pinv = PyObject_GetAttrString(linalg, "pinv");
    if (!m_pinv)
      return;
    m_p_.releaseAndSet(m_pinv);
    m_svd = PyObject_GetAttrString(linalg, "svd");
    if (!m_svd)
      return;
    m_s_.releaseAndSet(m_svd);
    m_ok = true;

    if (wantQR) {
      //For some reason I don't understand, I have to use the higher level
      //PyImport_Import to load scipy.linalg
      auto pckg_name = PyUnicode_FromString("scipy.linalg");
      RefHolder pckg_name_(pckg_name);
      PyObject* scipyLinalg = PyImport_Import(pckg_name);
      if (!scipyLinalg) {
        PyErr_Clear();
        return;
      }
      RefHolder scipyLinalg_(scipyLinalg);
      m_qrScipy = PyObject_GetAttrString(scipyLinalg, "qr");
      if (m_qrScipy)
        m_q_.releaseAndSet(m_qrScipy);
      else
        PyErr_Clear();
    }
  }
  PyObject* lstsqWithTranspose(PyObject *a1, PyObject *a2) const {
    PyObject* a1t = PyObject_CallFunctionObjArgs(m_transpose, a1, NULL);
    if (!a1t)
      return nullptr;
    RefHolder a1t_(a1t);
    PyObject* o = PyObject_CallFunctionObjArgs(m_lstsq, a1t, a2, NULL);
    if (!o)
      return nullptr;
    //return o;
    //return PyTuple_Pack(2,o,a1);
    RefHolder o_(o);
    PyObject* answer = PyTuple_GetItem(o, 0);
    Py_INCREF(answer);
    return answer;
  }
  PyObject* lstsqNoTranspose(PyObject *a1, PyObject *a2) const {
    PyObject* o = PyObject_CallFunctionObjArgs(m_lstsq, a1, a2, NULL);
    if (!o)
      return nullptr;
    //return o;
    //return PyTuple_Pack(2,o,a1);
    RefHolder o_(o);
    PyObject* answer = PyTuple_GetItem(o, 0);
    Py_INCREF(answer);
    return answer;
  }
  //a numpy matrix of an existing vector. Const correctness is user's responsibility
  template<class Type>
  class MatrixOfVector {
  public:
    PyObject* m_mat;
    MatrixOfVector(const vector<typename Type::T>& v, npy_intp r, npy_intp c) {
      npy_intp dims[] = { r,c };
      m_mat = PyArray_SimpleNewFromData(2, dims, Type::typenum, (void*)v.data());
    }
    MatrixOfVector(const MatrixOfVector&) = delete;
    MatrixOfVector operator=(const MatrixOfVector&) = delete;
    ~MatrixOfVector() { Py_DECREF(m_mat); }
  };

  //given an rxc matrix mat, find its rank and the first part of its SVD
  //on failure, m_ok will be false.
  //expect r>c, so that we are interested in the span of the columns,
  class SVD {
  public:
    double* m_u;//is rxr if full_matrices, else rxc
    bool m_ok = false;
    ReleasableRefHolder m_delete;
    int m_rank = 0;
    SVD(const vector<double>& mat, npy_intp r, npy_intp c, bool full_matrices, LeastSquares& ls) {
      MatrixOfVector<UseDouble> mat_(mat, r, c);
      if (mat.size() != ((size_t)r*c))
        ERRr("bad use of SVD");
#if 0
      for (int rr = 0, i = 0; rr < r; ++rr) {
        for (int cc = 0; cc < c; ++cc, ++i)
          std::cout << mat[i] << ",";
        std::cout << "\n";
      }
#endif
      PyObject* svd = PyObject_CallFunctionObjArgs(ls.m_svd, mat_.m_mat, 
        full_matrices ? Py_True : Py_False, NULL);
      if (!svd)
        return;
      m_delete.releaseAndSet(svd);
      PyObject* svs = PyTuple_GetItem(svd, 1);
      PyArrayObject* svd0 = (PyArrayObject*)PyTuple_GetItem(svd, 0);
      bool ok = PyArray_NDIM((PyArrayObject*)svs) == 1 &&
        PyArray_TYPE((PyArrayObject*)svs) == NPY_FLOAT64 &&
        PyArray_NDIM(svd0) == 2 &&
        PyArray_TYPE(svd0) == NPY_FLOAT64 &&
        PyArray_DIM(svd0, 0) == r &&
        PyArray_DIM(svd0, 1) == (full_matrices ? r : c) &&
        PyArray_ISCARRAY_RO(svd0) &&
        PyArray_ISCARRAY_RO((PyArrayObject*)svs)
        ;
      if (!ok)
        ERRr("numpy svd returned something strange.");
      npy_intp n_svs = PyArray_DIM((PyArrayObject*)svs, 0);
#ifdef PRINT_SVS
      std::cout << "\n";
#endif
      for (npy_intp i = 0; i < n_svs; ++i) {
        double sv = *((double*)PyArray_GETPTR1((PyArrayObject*)svs, i));
#ifdef PRINT_SVS
        std::cout << sv<<",";
#endif
        if (sv > 0.000001)
          ++m_rank;
      }
#ifdef PRINT_SVS
      std::cout << std::endl;
#endif
      m_u=(double*)PyArray_DATA(svd0);
      m_ok = true;
    }
  };

  //given an rxc matrix mat, find its rank and the Q of its QR
  //on failure, m_ok will be false.
  //We have r>c, and we are interested in the span of the columns.
  //So we need to use scipy's qr, which provides pivoting.
  //The user of this class must have already checked 
  //that m_qrScipy is not nullptr.
  class QR {
  public:
    double* m_q;//is rxc
    bool m_ok = false;
    ReleasableRefHolder m_delete;
    int m_rank = 0;
    QR(const vector<double>& mat, npy_intp r, npy_intp c, LeastSquares& ls) {
      MatrixOfVector<UseDouble> mat_(mat, r, c);
      if (mat.size() != ((size_t)r*c))
        ERRr("bad use of QR");
      char* Oissi = (char*) "Oissi"; //Yukky cast, but needed for older pythons
      PyObject* qr = PyObject_CallFunction(ls.m_qrScipy, Oissi, mat_.m_mat, 0,NULL,"economic",1);
      if (!qr)
        return;
      m_delete.releaseAndSet(qr);
      bool ok1 = PyTuple_CheckExact(qr) &&
        PyTuple_Size(qr) > 1 &&
        PyArray_Check(PyTuple_GetItem(qr, 0)) &&
        PyArray_Check(PyTuple_GetItem(qr, 1));
      if(!ok1)
        ERRr("scipy qr returned something very strange.");
      PyArrayObject* qr0 = (PyArrayObject*)PyTuple_GetItem(qr, 0);
      PyArrayObject* qr1 = (PyArrayObject*)PyTuple_GetItem(qr, 1);
      //NB: It seems scipy returns qr0 as fortran, numpy's qr doesnt
      bool fortran = PyArray_ISFARRAY_RO(qr0);
      bool qr0contig = fortran; 
      bool ok2 = PyArray_NDIM(qr0) == 2 &&
        PyArray_TYPE(qr0) == NPY_FLOAT64 &&
        PyArray_DIM(qr0, 0) == r &&
        PyArray_DIM(qr0, 1) == c &&
        PyArray_NDIM(qr1) == 2 &&
        PyArray_TYPE(qr1) == NPY_FLOAT64 &&
        PyArray_DIM(qr1, 0) == c &&
        PyArray_DIM(qr1, 1) == c &&
        qr0contig;
      if (!ok2)
        ERRr("scipy qr returned something strange.");
#ifdef PRINT_SVS
      std::cout << "\n";
#endif
      for (npy_intp i = 0; i < c; ++i) {
        double sv = *((double*)PyArray_GETPTR2((PyArrayObject*)qr1, i, i));
#ifdef PRINT_SVS
        std::cout << sv << ",";
#endif
        if (sv > 0.000001 || sv < -0.000001)
          ++m_rank;
      }
#ifdef PRINT_SVS
      std::cout << std::endl;
#endif
      m_q = (double*)PyArray_DATA(qr0);
      m_ok = true;
    }
  };

  //this replaces the r x c matrix matrix by its Moore-Penrose pseudoinverse, 
  //which is a c x r matrix 
  //returns true on success
  bool inplacePinvMatrix(float* matrix, size_t r, size_t c) const {
    vector<float> input(matrix, matrix + r*c);
    npy_intp dims2[] = { (npy_intp)(r), (npy_intp)c };
    PyObject* mat = PyArray_SimpleNewFromData(2, dims2, NPY_FLOAT32, input.data());
    RefHolder m_(mat);
    PyObject* o = PyObject_CallFunctionObjArgs(m_pinv, mat, NULL);
    if (!o)
      return false;
    RefHolder o_(o);
    PyArrayObject* oa = reinterpret_cast<PyArrayObject*>(o);
    bool ok = PyArray_Check(o) && PyArray_TYPE(oa) == NPY_FLOAT32
      && PyArray_NDIM(oa) == 2 
      && PyArray_DIM(oa, 0) == ((npy_intp)c) && PyArray_DIM(oa, 1) == ((npy_intp)r);
    if (!ok)
      ERRb("bad output from pinv");
    PyArrayObject* outc = PyArray_GETCONTIGUOUS(oa);
    RefHolder l2_(reinterpret_cast<PyObject*>(outc));
    float* ptr = static_cast<float*>(PyArray_DATA(outc));
    for (size_t i = 0; i < r*c; ++i)
      matrix[i] = ptr[i];
    return true;
  }

  //inp is a dxm matrix and sub is a dxn matrix with n<m
  //set out to a dx(roughly m-n) matrix whose columns are perp to colspan(sub) and each other
  //s.t. colspan(out)=colspan(inp)
  //This current method may be a big waste of time. We shouldn't need d^2 space.
  //I wonder if qr would beat svd, but I think we'd still need two.
  bool projectAwayFrom1(const vector<double>& inp, const vector<double>& sub, npy_intp d, vector<double>& out) {
    npy_intp m = inp.size() / d;
    npy_intp n = sub.size() / d;

    SVD svd(sub, d, n, true, *this);
    if (!svd.m_ok)
      return false;

    double* s1 = svd.m_u;
    int sub_rank = svd.m_rank;
    vector<double> inp_projected(d*m, 0);
    //std::cout << "!"<< d << "," << m << "," << n << "," << sub_rank << std::endl;
    //s1 is a dxd array. If s2 is s1 without its first sub_rank columns, 
    //then s3=s2 s2^T is a projection matrix away from sub
    //we want inp_projected= s3.inp = s2 . (s2^T.inp) =: s2.s4
    
    if (0) {
      vector<double> s3(d*d, 0);
      for (int i = 0; i < d; ++i) {
        for (int j = 0; j + sub_rank < d; ++j) {
          for (int k = 0; k < d; ++k) {
            s3.at(i*d + k) += s1[i*d + sub_rank + j] * s1[k*d + sub_rank + j];
          }
        }
      }

      for (int i = 0; i < m; ++i)
        for (int j = 0; j < d; ++j)
          for (int k = 0; k < d; ++k) {
            //if (i*m + k >= inp.size())
              //std::cout << "'" << inp.size()<<","<<s3.size()<<","<<inp_projected.size()
              //  << "," << m << "," << d << "," << i << "," << j << "," << k << std::endl;
            inp_projected.at(j*m + i) += inp.at(k*m + i) * s3.at(j*d + k);
          }
    }
    else { // group s.t. second multiply happens first
      npy_intp s4_rows = d - sub_rank;
      vector<double> s4((d - sub_rank)*d);
      for (npy_intp i = 0; i < d; ++i)
        for (npy_intp j = 0; j < m; ++j)
          for (npy_intp k = 0; k < s4_rows; ++k)
            s4.at(k*s4_rows + j) += s1[i*d + sub_rank + k] * inp.at(i*m+j);
      std::cout << "p4" << std::endl;
      for (npy_intp i = 0; i < d; ++i)
        for (npy_intp j = 0; j < m; ++j)
          for (npy_intp k = 0; k < s4_rows; ++k)
            inp_projected.at(i*m + j) += s1[i*d + sub_rank + k] * s4.at(k*s4_rows + j);
    }

    SVD svd2(inp_projected, d, m, false, *this); 
    if (!svd2.m_ok)
      return false;

    int out_rank = svd2.m_rank;
    out.assign(d*out_rank,0);
    for (npy_intp i = 0; i < d; ++i)
      for (int j = 0; j < out_rank; ++j) {
        //std::cout << i << "," << j << "," << d << "," << out_rank << std::endl;
        out.at(i*out_rank + j) = svd2.m_u[i*m + j];
      }
    return true;
  }

  //inp is a dxm matrix and sub is a dxn matrix with n<m
  //set out to a dx(roughly m-n) matrix whose columns are perp to colspan(sub) and each other
  //s.t. colspan(out)=colspan(inp)
  //This is like the previous only the projection matrix is not constructed.
  //space d^2 is never needed.
  bool projectAwayFromSVD(const vector<double>& inp, const vector<double>& sub, npy_intp d, vector<double>& out) {
    npy_intp m = inp.size() / d;
    npy_intp n = sub.size() / d;

    SVD svd(sub, d, n, false, *this);
    if (!svd.m_ok)
      return false;

    vector<double> inp_projected = inp;
    for (npy_intp i = 0; i < m; ++i)
      for (npy_intp j = 0; j < svd.m_rank; ++j) {
        double dot_product = 0;
        for (npy_intp k = 0; k < d; ++k)
          dot_product += inp.at(k*m + i)*svd.m_u[k*n + j];
        for (npy_intp k = 0; k < d; ++k)
          inp_projected[k*m + i] -= dot_product * svd.m_u[k*n + j];
      }

    SVD svd2(inp_projected, d, m, false, *this);
    if (!svd2.m_ok)
      return false;

    int out_rank = svd2.m_rank;
    out.assign(d*out_rank, 0);
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < out_rank; ++j) {
        //std::cout << i << "," << j << "," << d << "," << out_rank << std::endl;
        out.at(i*out_rank + j) = svd2.m_u[i*m + j];
      }
    return true;
  }
  //Like projectAwayFromSVD only using QR instead of SVD
  // - relies on scipy
  bool projectAwayFromQR(const vector<double>& inp, const vector<double>& sub, npy_intp d, vector<double>& out) {
    npy_intp m = inp.size() / d;
    npy_intp n = sub.size() / d;

    if (!m_qrScipy)
      ERRb("scipy not available, QR method cannot be used");

    QR qr1(sub, d, n, *this);
    if (!qr1.m_ok)
      return false;

    vector<double> inp_projected = inp;
    for (npy_intp i = 0; i < m; ++i)
      for (npy_intp j = 0; j < qr1.m_rank; ++j) {
        double dot_product = 0;
        for (npy_intp k = 0; k < d; ++k)
          dot_product += inp.at(k*m + i)*qr1.m_q[k + j*d];
        for (npy_intp k = 0; k < d; ++k)
          inp_projected[k*m + i] -= dot_product * qr1.m_q[k + j*d];
      }

    QR qr2(inp_projected, d, m, *this);
    if (!qr2.m_ok)
      return false;

    int out_rank = qr2.m_rank;
    out.assign(d*out_rank, 0);
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < out_rank; ++j) {
        //std::cout << i << "," << j << "," << d << "," << out_rank << std::endl;
        out.at(i*out_rank + j) = qr2.m_q[i + j*d];
      }
    return true;
  }
};

//things which have to be done at the end of prepare, relying on Numpy
//returns false on error
//can also be called in logsigbackwards
bool postPrepare(LogSigFunction* lsf) {
  LeastSquares ls;
  if (!ls.m_ok)
    return false;
  for (auto& i : lsf->m_smallSVDs)
    for (auto& j : i) {
      if (j.m_matrix.empty())
        continue;
      bool ok = ls.inplacePinvMatrix(j.m_matrix.data(),
        j.m_sources.size(), j.m_dests.size());
      if (!ok)
        return false;
    }
  return true;
}
#endif // IISIGNATURE_NO_NUMPY

//This function takes a dim1xdim2 matrix and an rhs and calls resultAction on a pointer
//to the results of lstsq on it
//returns true on success
/*
template<typename T>
bool callLeastSquares(LeastSquares& ls, float* matrix, size_t dim1, size_t dim2, float* rhs,
  bool transpose, T&& resultAction) {
  const size_t rhs_length = (transpose ? dim2 : dim1);
  //const size_t out_length = (transpose ? dim1 : dim2);
  npy_intp dims[] = { (npy_intp)rhs_length };
  PyObject* rhs_ = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, rhs);
  RefHolder r_(rhs_);

  npy_intp dims2[] = { (npy_intp)(dim1), (npy_intp)dim2 };
  PyObject* mat = PyArray_SimpleNewFromData(2, dims2, NPY_FLOAT32, matrix);
  RefHolder m_(mat);
  PyObject* out = transpose ? ls.lstsqWithTranspose(mat, rhs_) : ls.lstsqNoTranspose(mat, rhs_);
  if (!out)
    return false;
  RefHolder o_(out);
  auto outa = reinterpret_cast<PyArrayObject*>(out);
  PyArrayObject* outc = PyArray_GETCONTIGUOUS(outa);
  RefHolder l2_(reinterpret_cast<PyObject*>(outc));
  float* ptr = static_cast<float*>(PyArray_DATA(outc));
  if (false) {
    for (size_t row = 0;row < dim1;row++) {
      for (size_t col = 0; col < dim2; ++col)
        std::cout << matrix[row*dim2 + col] << ",";
      std::cout << "\n";
    }
    std::cout << (transpose ? "transpose times\n" : "times\n");
    for (size_t row = 0; row < out_length;++row)
      std::cout << ptr[row] << "\n";
    std::cout << "is\n";
    for (size_t row = 0; row<rhs_length;++row)
      std::cout <<rhs[row] << "\n";
    std::cout << std::endl;
  }
  resultAction(ptr);
  return true;
}
*/

static PyObject *
prepare(PyObject *self, PyObject *args){
  int level=0, dim=0;
  const char* methods = nullptr;
  if (!PyArg_ParseTuple(args, "ii|z", &dim, &level, &methods))
    return nullptr;
  if(!getData())
    return nullptr;
  if(dim<2) ERR("dimension must be at least 2");
  if(level<1) ERR("level must be positive");
  WantedMethods wantedmethods;
  std::string methodString;
  if(methods)
    methodString = methods;
  if(setWantedMethods(wantedmethods,dim,level,false,methodString))
    ERR(wantedmethods.m_errMsg);
  auto basis = wantedmethods.m_want_matchCoropa ? LieBasis::StandardHall : LieBasis::Lyndon;
  std::unique_ptr<LogSigFunction> lsf(new LogSigFunction(basis));
  bool ok = do_interruptible_releasing_lock([&]{
    makeLogSigFunction(dim,level,*lsf, wantedmethods, interrupt);
    return true;
  });
  if (!ok)
    return nullptr;

#ifndef IISIGNATURE_NO_NUMPY
  if (!postPrepare(lsf.get()))
    return nullptr;
#endif // IISIGNATURE_NO_NUMPY

#ifdef NO_CAPSULES
  PyObject * out = PyCObject_FromVoidPtr(lsf.release(), killLogSigFunction);
#else
  PyObject * out = PyCapsule_New(lsf.release(), logSigFunction_id, killLogSigFunction);
#endif
  return out;
}

static PyObject* basis(PyObject *self, PyObject *args){
  PyObject* c;
  if(!PyArg_ParseTuple(args,"O",&c))
    return nullptr;
  LogSigFunction* lsf = getLogSigFunction(c);
  if(!lsf)
    return nullptr;
  auto& elementList = lsf->m_basisElements;
  PyObject* o = PyTuple_New(elementList.size());
  if (!o)
    return nullptr;
  for(size_t i=0; i<elementList.size(); ++i){
    std::ostringstream oss;
    printBasisEltBracketsDigits(*elementList[i],oss);
    std::string s = oss.str();
    PyTuple_SET_ITEM(o,i,PyUnicode_FromString(s.c_str()));
  }
  return o;
}

static PyObject* info(PyObject *self, PyObject *args) {
  PyObject* c;
  if (!PyArg_ParseTuple(args, "O", &c))
    return nullptr;
  LogSigFunction* lsf = getLogSigFunction(c);
  if (!lsf)
    return nullptr;
  std::string methods;
  if (lsf->m_f)
    methods += 'C';
  if (!lsf->m_fd.m_formingT.empty())
    methods += 'O';
  if (lsf->canProjectToBasis())
    methods += 'S';
  methods += 'X';
  const char* basis = (lsf->m_s.m_basis == LieBasis::StandardHall) ? 
                          "Standard Hall" : "Lyndon";
  return Py_BuildValue("{sisissss}", "dimension", lsf->m_dim, "level",
    lsf->m_level, "methods", methods.c_str(), "basis", basis);
}

static int getSigLength(LogSigFunction* lsf) {
  if (lsf->m_siglength != 0)
    return lsf->m_siglength;
  return calcSigTotalLength(lsf->m_dim, lsf->m_level);
}

#ifndef IISIGNATURE_NO_NUMPY
static PyObject *
logsig(PyObject *self, PyObject *args){
  PyObject* a1, *a2;
  const char* methods = nullptr;
  if (!PyArg_ParseTuple(args, "OO|z", &a1, &a2, &methods))
    return nullptr;
  LogSigFunction* lsf = getLogSigFunction(a2);
  if(!lsf)
    return nullptr;
  WantedMethods wantedmethods;
  std::string methodString;
  if(methods)
    methodString = methods;
  if(setWantedMethods(wantedmethods,lsf->m_dim,lsf->m_level,true,methodString))
    ERR(wantedmethods.m_errMsg);
  PyObject* aa = PyArray_ContiguousFromAny(a1, NPY_FLOAT64, 0, 0);
  if (!aa) ERR("data must be (convertable to) a numpy array");
  RefHolder a_(aa);
  PyArrayObject* a = (PyArrayObject*)aa;

  int ndims = PyArray_NDIM(a);
  if (ndims < 2) ERR("data must be 2d");
  const int lengthOfPath = (int)PyArray_DIM(a, ndims - 2);
  const int d = (int)PyArray_DIM(a, ndims - 1);
  const size_t eachInputLength = (size_t)(lengthOfPath * d);
  if (lengthOfPath < 1) ERR("Path has no length");
  if (d != lsf->m_dim)
    ERR(("Path has dimension " + std::to_string(d) + " but we prepared for dimension "
      + std::to_string(lsf->m_dim)).c_str());
  size_t logsiglength = lsf->m_basisElements.size();
  double* data = static_cast<double*>(PyArray_DATA(a));

  int nPaths = 1;
  for (int i = 0; i + 2 < ndims; ++i) {
    npy_intp x = PyArray_DIM(a, i);
    nPaths *= (int)x;
  }

  FunctionRunner* f = lsf->m_f.get();
  using OutT=UseDouble;
  if ((wantedmethods.m_compiled_bch && f!=nullptr) || 
      (wantedmethods.m_simple_bch && lsf->m_fd.m_length_of_b>0)){
    vector<double> displacement(d);
    const bool useCompiled = (f!=nullptr && wantedmethods.m_compiled_bch);

    PyObject* o = simpleNew_ownLastDim(ndims - 1, a, logsiglength, OutT::typenum);
    if (!o)
      return nullptr;
    auto dest = static_cast<OutT::T*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o)));
    vector<double> out;
    for (int path = 0; path < nPaths; ++path) {
      out.assign(logsiglength, 0);
      if (lengthOfPath > 1) {
        for (int j = 0; j < d; ++j)
          out[j] = data[1 * d + j] - data[0 * d + j];
      }
      for (int i = 2; i < lengthOfPath; ++i) {
        for (int j = 0; j < d; ++j)
          displacement[j] = data[i*d + j] - data[(i - 1)*d + j];
        if (useCompiled)
          f->go(out.data(), displacement.data());
        else
          slowExplicitFunction(out.data(), displacement.data(), lsf->m_fd);
      }
      for (double d : out)
        *dest++ = (OutT::T) d;
      data += eachInputLength;
    }
    return o;
  }
  if (wantedmethods.m_expanded ||
    (wantedmethods.m_log_of_signature && lsf->canProjectToBasis())) {
    size_t eachOutputSize = wantedmethods.m_expanded ?
      (size_t)getSigLength(lsf) : (npy_intp)logsiglength;
    PyObject* o = simpleNew_ownLastDim(ndims - 1, a, eachOutputSize, OutT::typenum);
    if (!o)
      return nullptr;
    auto outp = static_cast<OutT::T*>(PyArray_DATA(
      reinterpret_cast<PyArrayObject*>(o)));
    ReleasableRefHolder o_(o);

    Signature sig;
    bool ok = do_interruptible([&] {
      for (int path = 0; path < nPaths; ++path) {
        if (!calcSignature(sig, data, lengthOfPath, d, lsf->m_level))
          return false;
        data += eachInputLength;
        logTensorHorner(sig);
        if (wantedmethods.m_expanded)
          sig.writeOut(outp + path * eachOutputSize);
        else {
          projectExpandedLogSigToBasis(outp + path * eachOutputSize,
            lsf, sig);
        }
      }
      return true;
    });
    if (!ok)
      return nullptr;
    o_.release();
    return o;
  }
  ERR("We had not prepare()d for this request type");
}

static PyObject *
logsigbackwards(PyObject *self, PyObject *args) {
  PyObject *a0, *a1, *a2;
  const char* methods = nullptr;
  if (!PyArg_ParseTuple(args, "OOO|z", &a0, &a1, &a2, &methods))
    return nullptr;
  LogSigFunction* lsf = getLogSigFunction(a2);
  if (!lsf)
    return nullptr;

  WantedMethods wantedmethods;
  std::string methodString;
  if (methods)
    methodString = methods;
  if (setWantedMethods(wantedmethods, 100, 100, true, methodString))
    ERR(wantedmethods.m_errMsg);
  if (!(wantedmethods.m_log_of_signature || wantedmethods.m_expanded))
    ERR("Requested method cannot be used here");
  bool needProjectionButDontHaveIt = !wantedmethods.m_expanded && !lsf->canProjectToBasis();
  //if (needProjectionButDontHaveIt)
  //  ERR("We had not prepared the S method");

  PyObject* aa = PyArray_ContiguousFromAny(a1, NPY_FLOAT64, 0, 0);
  if (!aa) ERR("data must be (convertable to) a numpy array");
  RefHolder a_(aa);
  PyArrayObject* a = (PyArrayObject*)aa;

  PyObject* derivs = PyArray_ContiguousFromAny(a0, NPY_FLOAT64, 0, 0);
  if (!derivs) ERR("derivs must be (convertable to) a numpy array");
  RefHolder derivs_(derivs);
  PyArrayObject* derivsa = (PyArrayObject*)derivs;

  int ndims = PyArray_NDIM(a);
  if (ndims < 2) ERR("data must be at least 2d");
  const int lengthOfPath = (int)PyArray_DIM(a, ndims - 2);
  const int d = (int)PyArray_DIM(a, ndims - 1);
  const size_t eachInputLength = (size_t)(lengthOfPath * d);
  if (lengthOfPath < 1) ERR("Path has no length");
  if (d != lsf->m_dim)
    ERR(("Path has dimension " + std::to_string(d) + " but we prepared for dimension "
      + std::to_string(lsf->m_dim)).c_str());
  if (PyArray_NDIM(derivsa) != ndims - 1) ERR("derivs has the wrong number of dimensions");
  size_t logsiglength = lsf->m_basisElements.size();
  const double* data = static_cast<double*>(PyArray_DATA(a));
  const double* derivPtr = static_cast<double*>(PyArray_DATA(derivsa));

  int nPaths = 1;
  for (int i = 0; i + 2 < ndims; ++i) {
    npy_intp x = PyArray_DIM(a, i);
    if (PyArray_DIM(derivsa, i) != x)
      ERR("mismatched dimensions between sigs and derivs");
    nPaths *= (int)x;
  }

  using OutT = UseFloat;
  size_t eachOutputSize = wantedmethods.m_expanded ?
    (size_t)getSigLength(lsf) : (npy_intp)logsiglength;
  if (eachOutputSize != (size_t) PyArray_DIM(derivsa, ndims - 2))
    ERR("derivatives have unexpected length");
  PyObject* o = PyArray_ZEROS(ndims, PyArray_DIMS(a), OutT::typenum, 0);
  if (!o)
    return nullptr;
  auto outp = static_cast<OutT::T*>(PyArray_DATA(
    reinterpret_cast<PyArrayObject*>(o)));
  ReleasableRefHolder o_(o);

  Signature sig, sigDer;
  if (needProjectionButDontHaveIt) {
    bool ok = do_interruptible([&] {
      IISignature_algebra::makeSparseLogSigMatrices(lsf->m_dim, lsf->m_level, *lsf, interrupt);
      return true;
    });
    ok = ok && postPrepare(lsf);
    if (!ok) {
      //We've failed to add projection info to lsf, we better leave lsf in its safe, original state
      lsf->m_simples.clear();
      lsf->m_smallSVDs.clear();
      lsf->m_smallTriangles.clear();
      return nullptr;
    }
  }
  bool ok = do_interruptible([&] {
    for (int path = 0; path < nPaths; ++path) {
      calcSignature(sig, data, lengthOfPath, d, lsf->m_level);
      if (wantedmethods.m_expanded)
        sigDer.fromRaw(lsf->m_dim, lsf->m_level, derivPtr);
      else
        projectExpandedLogSigToBasisBackwards(derivPtr, lsf, sigDer);
      derivPtr += eachOutputSize;
      logBackwards(sigDer, sig);
      //This should use the calculated signature
      CalcSignature::sigBackwards(lsf->m_dim, lsf->m_level, lengthOfPath, data, sigDer, outp + path*eachInputLength);
      data += eachInputLength;
    }
    return true;
  });
  if (!ok)
    return nullptr;
  o_.release();
  return o;
}

#endif

const char* const rotInv_id = "iisignature.rotInv";
RotationalInvariants::Prepared* getPreparedRotInv(PyObject* p) {
#ifdef NO_CAPSULES
  if (!PyCObject_Check(p))
    ERR("I have received an input which is not from iisignature.rotinv2dprepare()");
  return (RotationalInvariants::Prepared*)PyCObject_AsVoidPtr(p);
#else
  if (!PyCapsule_IsValid(p,rotInv_id))
    ERR("I have received an input which is not from iisignature.rotinv2dprepare()");
  return (RotationalInvariants::Prepared*)PyCapsule_GetPointer(p, rotInv_id);
#endif
}

#ifndef NO_CAPSULES
static void killRotInv(PyObject* p) {
#else
static void killRotInv(void* v) {
  PyObject* p = (PyObject*)v;
#endif
  delete getPreparedRotInv(p);
}

static PyObject *
rotinv2dlength(PyObject *self, PyObject *args) {
/*
  int level = 0;
  if (!PyArg_ParseTuple(args, "i", &level))
    return nullptr;
  if (level < 2 || level > 63 || level % 2 != 0)
    ERR("Level must be a small positive even number");
  long ans = 0;
  for (int lev = 2; lev <= level; lev += 2)
    //length is (lev choose lev/2)
    ans += (long)LogSigLength::centralBinomialCoefficient(lev);*/
  PyObject* a1;
  if (!PyArg_ParseTuple(args, "O", &a1))
    return nullptr;
  auto prepared = getPreparedRotInv(a1);
  if (!prepared)
    return nullptr;
  long ans = 0;
  for (const auto& i : prepared->m_invariants)
    ans += (long)i.size();

#if PY_MAJOR_VERSION >= 3
  return PyLong_FromLong(ans);
#else
  return PyInt_FromLong(ans);
#endif
}

static PyObject *
rotinv2dprepare(PyObject *self, PyObject *args) {
  int level = 0;
  const char* type = nullptr;
  if (!PyArg_ParseTuple(args, "iz", &level, &type))
    return nullptr;
  if (level < 2 || level > 63 || level % 2 != 0)
    ERR("Level must be a small positive even number");
  RotationalInvariants::InvariantType t;
  if (type) {
    if (RotationalInvariants::getWantedMethod(type, t))
      ERR("Invalid type of rotational invariant")
  }
  else
    t = RotationalInvariants::InvariantType::SVD;

  using T = RotationalInvariants::Prepared;
  auto p = std::unique_ptr<T>(new T(level,t));
#ifndef IISIGNATURE_NO_NUMPY
  if (t == RotationalInvariants::InvariantType::SVD || 
      t==RotationalInvariants::InvariantType::QR) {
    LeastSquares ls (t == RotationalInvariants::InvariantType::QR);
    if (!ls.m_ok)
      return nullptr;
    vector<double> tempa, tempb, tempc;
    for (int lev = 4; lev <= level; lev += 2) {
      for (int parity : {0, 1}) {
        size_t idx = lev - 2 + parity;
        RotationalInvariants::invariantsToMatrix(p->m_invariants[idx], lev, tempa);
        RotationalInvariants::invariantsToMatrix(p->m_knownInvariants[idx], lev, tempb);
        bool ok = false;
        if (t == RotationalInvariants::InvariantType::QR)
          ok = ls.projectAwayFromQR(tempa, tempb, ((npy_intp)1) << (lev - 1), tempc);
        else
          ok = ls.projectAwayFromSVD(tempa, tempb, ((npy_intp)1) << (lev - 1), tempc);
        if (!ok) 
          return nullptr;
        RotationalInvariants::invariantsFromMatrix(tempc, lev, parity, p->m_invariants[idx]);
      }
    }
  }
#endif
  if (t == RotationalInvariants::InvariantType::KNOWN)
    p->m_invariants = p->m_knownInvariants;
#ifdef NO_CAPSULES
  PyObject * out = PyCObject_FromVoidPtr(p.release(), killRotInv);
#else
  PyObject * out = PyCapsule_New(p.release(), rotInv_id, killRotInv);
#endif
  return out;
}

#ifndef IISIGNATURE_NO_NUMPY
//iisignature.rotinv2d(numpy.random.uniform(size=(12,2)),4)
static PyObject *
rotinv2d(PyObject *self, PyObject *args) {
  PyObject* a1, *a2;
  if (!PyArg_ParseTuple(args, "OO", &a1, &a2))
    return nullptr;
  auto prepared = getPreparedRotInv(a2);
  if (!prepared)
    return nullptr;
  int level = prepared->m_level;
  size_t nInvariants = 0;
  for (const auto& i : prepared->m_invariants)
    nInvariants += (long)i.size();

  PyObject* aa = PyArray_ContiguousFromAny(a1, NPY_FLOAT64, 0, 0);
  if (!aa) ERR("data must be (convertable to) a numpy array");
  RefHolder a_(aa);
  PyArrayObject* a = (PyArrayObject*)aa;
  int ndims = PyArray_NDIM(a);
  if (ndims < 2) ERR("data must be 2d");
  const int lengthOfPath = (int)PyArray_DIM(a, ndims - 2);
  const int d = (int)PyArray_DIM(a, ndims - 1);
  if (lengthOfPath < 1) ERR("Path has no length");
  if (d != 2)
    ERR(("Path has dimension " + std::to_string(d) + " but must have dimension 2.").c_str());
  if (level < 2 || level > 63 || level % 2 != 0)
    ERR("Level must be a small positive even number");
  int nPaths = 1;
  for (int i = 0; i + 2 < ndims; ++i) {
    npy_intp x = PyArray_DIM(a, i);
    nPaths *= (int)x;
  }
  size_t eachInputSize = (size_t)(lengthOfPath * d);
  using OutT = UseDouble;
  PyObject* o = simpleNew_ownLastDim(ndims - 1, a, nInvariants, OutT::typenum);
  if (!o)
    return nullptr;
  ReleasableRefHolder o_(o);
  OutT::T* out_data = static_cast<OutT::T*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o)));
  auto in_data = (double*)PyArray_DATA(a);

  Signature sig;
  std::vector<double> out(nInvariants);
  bool ok = do_interruptible([&] {
    for (int path = 0; path < nPaths; ++path) {
      if (!calcSignature(sig, in_data, lengthOfPath, d, level))
        return false;
      for (int lev = 2; lev <= level; lev += 2) {
        for (int parity : {0, 1}) {
          size_t idx = lev - 2 + parity;
          auto& invs = prepared->m_invariants[idx];
          for (auto& i : invs) {
            double d = 0;
            for (auto& p : i) {
              d += p.second*sig.m_data[lev - 1][(size_t)(p.first)];
            }
            *(out_data++) = d;
          }
        }
      }
      in_data += eachInputSize;
    }
    return true;
  });
  if (!ok)
    return nullptr;

  o_.release();
  return o;
}

static PyObject *
rotinv2dcoeffs(PyObject *self, PyObject *args) {
  PyObject* a1;
  if (!PyArg_ParseTuple(args, "O", &a1))
    return nullptr;
  auto prepared = getPreparedRotInv(a1);
  if (!prepared)
    return nullptr;
  int level = prepared->m_level;

  PyObject* out = PyTuple_New(level / 2);
  using OutT=UseDouble;
  for (int lev = 2; lev <= level; lev += 2) {
    auto& source0 = prepared->m_invariants[lev - 2];
    auto& source1 = prepared->m_invariants[lev - 1];
    size_t sourceSize = source0.size() + source1.size();
    size_t length = ((size_t)1u) << lev;
    npy_intp dims[] = { (npy_intp)(sourceSize), (npy_intp)length };
    PyObject* o = PyArray_SimpleNew(2, dims, OutT::typenum);
    if (!o)
      return nullptr; //but leak out...
    OutT::T* dest = static_cast<OutT::T*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o)));
    for (size_t i = 0; i < length*sourceSize; ++i)
      dest[i] = 0;
    for (size_t i = 0; i<source0.size(); ++i)
      for (const auto& p : source0[i])
        dest[i*length + p.first] = p.second;
    for (size_t i = 0; i<source1.size(); ++i)
      for (const auto& p : source1[i])
        dest[(i+source0.size())*length + p.first] = p.second;
    PyTuple_SET_ITEM(out, lev / 2 - 1, o);
  }
  
  return out;
}
#endif //IISIGNATURE_NO_NUMPY

#define METHOD_DESC "some combination of 'd' (the default), "\
  "'c' (the bch formula compiled on the fly), "\
  "'o' (the bch formula evaluated simply and stored in an object without " \
  "on-the-fly compilation and perhaps more slowly), "\
  "and 's' (calculating the log signature by first calculating "\
  "the signature and then taking its log, "\
  "which may be faster for high levels or long paths)"

static PyMethodDef Methods[] = {
#ifndef IISIGNATURE_NO_NUMPY
  {"sig",  sig, METH_VARARGS, "sig(X,m,format=1)\n Returns the signature of a path X "
   "up to level m. X must be a numpy [...x]NxD float32 or float64 array of points "
   "making up the path in R^d. The initial 1 in the zeroth level of the signature is excluded. "
   "If format is 1, the output is a list of arrays not a single one."},
  {"sigmultcount", sigMultCount, METH_VARARGS, "sigmultcount(X,m)\n "
   "Returns the number of multiplications which sig(X,m) would perform."},
  {"sigjacobian", sigJacobian, METH_VARARGS, "sigjacobian(X,m)\n "
   "Returns the full Jacobian matrix of "
   "derivatives of sig(X,m) with respect to X. "
   "If X is an NxD array then the output is an NxDx(siglength(D,m)) array."},
  {"sigbackprop", sigBackwards, METH_VARARGS, "sigbackprop(s,X,m)\n "
   "If s is the derivative of something with respect to sig(X,m), "
   "then this returns the derivative of that thing with respect to X. "
   "sigbackprop(s,X,m) should be approximately numpy.dot(sigjacobian(X,m),s)"},
  {"sigjoin", sigJoin, METH_VARARGS, "sigjoin(X,D,m,f=float('nan'))\n "
   "If X is an array of signatures of d dimensional paths of shape "
   "(..., siglength(d,m)) and D is an array of d dimensional displacements "
   "of shape (..., d), then return an array shaped like X "
   "of the signatures of the paths concatenated with the displacements. "
   "If f is provided, then it is taken to be the fixed value of the "
   "displacement in the last dimension, and D should have shape (K, d-1)."},
  {"sigjoinbackprop",sigJoinBackwards,METH_VARARGS, "sigjoinbackprop(s,X,D,m,f=float('nan')) \n "
   "gives the derivatives of F with respect to X and D (and f if given) where s is the derivatives"
   " of F with respect to sigjoin(X,D,m,f). The result is a tuple of two or three items."},
  {"sigscale", sigScale, METH_VARARGS, "sigjoin(X,D,m))\n "
   "If X is an array of signatures of d dimensional paths of shape "
   "(..., siglength(d,m)) and D is an array of d dimensional scales "
   "of shape (..., d), then return an array shaped like X "
   "of the signatures of the paths scaled by the corresponding scale factor in each dimension. "},
  {"sigscalebackprop",sigScaleBackwards,METH_VARARGS, "sigscalebackprop(s,X,D,m) \n "
   "gives the derivatives of F with respect to X and D where s is the derivatives"
   " of F with respect to sigscale(X,D,m). The result is a tuple of two items."},
#endif
  {"siglength", siglength, METH_VARARGS, "siglength(d,m) \n "
   "Returns the length of the signature (excluding the initial 1) of a d dimensional path up to level m"},
  {"logsiglength", logsiglength, METH_VARARGS, "logsiglength(d,m) \n "
   "Returns the length of the log signature of a d dimensional path up to level m"},
  {"rotinv2dlength", rotinv2dlength, METH_VARARGS, "rotinv2dlength(s) \n "
   "Returns the number of linear rotational invariants which rotinv2d(X,s) will return. "
   "s must be the result of a call to rotinv2dprepare."},
  {"rotinv2dprepare", rotinv2dprepare, METH_VARARGS, "rotinv2dprepare(m, type) \n "
   "This prepares the way to find linear rotational invariants of signatures up to level m "
   " of 2d paths. m must be even. The returned object is used in rotinv2d. type should be "
   "'a' to return all invariants, or 's' to use SVD to exclude those which are determined by "
   "shuffle products of lower level invariants."},
  {"prepare", prepare, METH_VARARGS, "prepare(d, m, methods=None) \n "
   "This prepares the way to calculate log signatures of d dimensional paths"
   " up to level m. The returned object is used in the logsig, basis and info functions. \n"
   " By default, all methods will be prepared, but you can restrict it "
   "by setting methods to " METHOD_DESC "."},
  {"basis", basis, METH_VARARGS, "basis(s) \n  Returns a tuple of strings "
   "which are the basis elements of the log signature. s must have come from prepare."
   " This function is work in progress, especially for dimension greater than 8. "
   "An example of how to parse the output of this function can be seen in the tests."},
  {"info", info, METH_VARARGS, "info(s) \n  Returns a dictionary of "
   "information about the opaque object s. s must have come from prepare."},
#ifndef IISIGNATURE_NO_NUMPY
  {"logsig", logsig, METH_VARARGS, "logsig(X, s, methods=None) \n "
   "Calculates the log signature of the path X. X must be a numpy [...x]Nxd float32 "
   "or float64 array of points making up the path in R^d. s must have come from "
   "prepare(d,m) for some m. The value is returned as a 1D numpy array of its "
   "log signature up to level m. By default, the method used is the default out "
   "of those which have been prepared, "
   "but you can restrict it by setting methods to " METHOD_DESC "."},
  {"logsigbackprop", logsigbackwards, METH_VARARGS, "logsigbackprop(y, X, s, methods=None) \n"
   "gives the derivatives of F with respect to X where y is the derivatives"
   " of F with respect to logsig(X,s,methods). The result has the same shape as X." },
  { "rotinv2d", rotinv2d, METH_VARARGS, "rotinv(X,s) \n "
  "Calculates the linear rotational invariants of the signature of the path X. X must be a "
  "[...x]Nx2 array of points making up the path(s) in R^2. "
  "The value is returned as a 1D numpy array "
  "of invariants from the signature. s must be the result of a call to rotinv2dprepare."},
  { "rotinv2dcoeffs", rotinv2dcoeffs, METH_VARARGS, "rotinv2dcoeffs(s) \n "
  "Returns the linear rotational invariants of the signature of the path X. X must be a numpy Nx2 float32 "
  "or float64 array of points making up the path in R^2. s must be the result of a call to rotinv2dprepare."
  "The value is returned as a list of 2D numpy arrays - (nInvariants x length) for each even level."},
#endif
  {"version", version, METH_NOARGS, "return the iisignature version string"},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

#define MODULEDOC "iisignature: Iterated integral signature and logsignature calculations"\
"\n\nPlease find documentation at http://www2.warwick.ac.uk/jreizenstein"\
" and code at https://github.com/bottler/iisignature."

#if PY_MAJOR_VERSION >= 3
 static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
    "iisignature",     /* m_name */
    MODULEDOC,  /* m_doc */
    -1,                  /* m_size */
    Methods,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
    };

PyMODINIT_FUNC
PyInit_iisignature(void){
#ifndef IISIGNATURE_NO_NUMPY
  import_array();
#endif
  PyObject* moduleObj =  PyModule_Create(&moduledef);
  PyModule_AddStringConstant(moduleObj, "__version__", TOSTRING(VERSION));
  return moduleObj;
}
#else

/*extern "C" __attribute__ ((visibility ("default"))) void */
PyMODINIT_FUNC
initiisignature(void)
{
#ifndef IISIGNATURE_NO_NUMPY
  import_array();
#endif
  PyObject* moduleObj = Py_InitModule3("iisignature", Methods, MODULEDOC);
  PyModule_AddStringConstant(moduleObj, "__version__", TOSTRING(VERSION));
}

#endif

/*
thinking about builds:
According to
http://python-packaging-user-guide.readthedocs.io/en/latest/distributing/#requirements-for-packaging-and-distributing
we can't distribute platform wheels on linux.
Basically we can only distribute source.
 Therefore we won't make wheels

 Build just with 
python setup.py sdist

*/
