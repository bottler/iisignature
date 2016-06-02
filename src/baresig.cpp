#include<utility>
#include<vector>
#include<utility>
#include<iostream>
#include<memory>
#include<limits>
#include<Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<numpy/arrayobject.h>

#include"calcSignature.h"
#include"logSigLength.h"


//This is a python addin to calculate signatures which doesn't use boost python - to be as widely buildable as possible.
//It should be buildable on tinis.

#define ERR(X) {PyErr_SetString(PyExc_RuntimeError,X); return NULL;}

class Deleter{
  PyObject* m_p;
public:
  Deleter(PyObject* p):m_p(p){};
  Deleter(const Deleter&)=delete;
  Deleter operator=(const Deleter&) = delete;
  ~Deleter(){Py_DECREF(m_p);}
};

static PyObject *
siglength(PyObject *self, PyObject *args){
  int d=0, m=0;
  if (!PyArg_ParseTuple(args, "ii", &d, &m))
    return NULL;
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
    return NULL;
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


static PyObject *
sig(PyObject *self, PyObject *args){
  PyObject* a1;
  int level=0;
  if (!PyArg_ParseTuple(args, "Oi", &a1, &level))
    return NULL;
  if(level<1) ERR("level must be positive");
  if(!PyArray_Check(a1)) ERR("data must be a numpy array");
  //PyArrayObject* a = reinterpret_cast<PyArrayObject*>(a1);
  PyArrayObject* a = PyArray_GETCONTIGUOUS(reinterpret_cast<PyArrayObject*>(a1));
  Deleter a_(reinterpret_cast<PyObject*>(a));
  if(PyArray_NDIM(a)!=2) ERR("data must be 2d");
  if(PyArray_TYPE(a)!=NPY_FLOAT32 && PyArray_TYPE(a)!=NPY_FLOAT64) ERR("data must be float32 or float64");
  const int lengthOfPath = PyArray_DIM(a,0);
  const int d = PyArray_DIM(a,1);
  if(lengthOfPath<1) ERR("Path has no length");
  if(d<1) ERR("Path must have positive dimension");
  CalculatedSignature s1, s2;
  vector<float> displacement(d);

  if(lengthOfPath==1){
    s2.sigOfNothing(d,level);
  }

  if(PyArray_TYPE(a)==NPY_FLOAT32){
    float* data = static_cast<float*>(PyArray_DATA(a));
    for(int i=1; i<lengthOfPath; ++i){
      for(int j=0;j<d; ++j)
        displacement[j]=data[i*d+j]-data[(i-1)*d+j];
      s1.sigOfSegment(d,level,&displacement[0]);
      if(i==1)
	s2.swap(s1);
      else
	s2.concatenateWith(d,level,s1);
    }
  }else{
    double* data = static_cast<double*>(PyArray_DATA(a));
    for(int i=1; i<lengthOfPath; ++i){
      for(int j=0;j<d; ++j)
        displacement[j]=data[i*d+j]-data[(i-1)*d+j];
      s1.sigOfSegment(d,level,&displacement[0]);
      if(i==1)
	s2.swap(s1);
      else
	s2.concatenateWith(d,level,s1);
    }
  }
    
  long dims[] = {(long) calcSigTotalLength(d,level)};
  PyObject* o = PyArray_SimpleNew(1,dims,NPY_FLOAT32);
  s2.writeOut(static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(o))));
  return o;
}

static PyMethodDef Methods[] = {
  {"sig",  sig, METH_VARARGS, "if X is a numpy NxD float32 or float64 array of points making a path in R^D then sig(X,m) returns a 1D numpy array of its signature up to level m, excluding the initial 1"},
  {"siglength", siglength, METH_VARARGS, "siglength(d,m) returns the length of the signature (excluding the initial 1) of a d dimensional path up to level m"},
  {"logsiglength", logsiglength, METH_VARARGS, "logsiglength(d,m) returns the length of the log signature of a d dimensional path up to level m"},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

#define MODULEDOC "Iterated integral signature and logsignature calculations"

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
  import_array();
  return PyModule_Create(&moduledef);
}
#else

/*extern "C" __attribute__ ((visibility ("default"))) void */
PyMODINIT_FUNC
initiisignature(void)
{
  import_array();
  (void) Py_InitModule3("iisignature", Methods, MODULEDOC);
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
