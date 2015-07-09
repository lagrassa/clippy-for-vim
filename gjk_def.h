#ifndef __PYX_HAVE__gjk
#define __PYX_HAVE__gjk


#ifndef __PYX_HAVE_API__gjk

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

__PYX_EXTERN_C DL_IMPORT(double) fizz(int);
__PYX_EXTERN_C DL_IMPORT(double) fuzz(int, int __pyx_skip_dispatch);

#endif /* !__PYX_HAVE_API__gjk */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initgjk(void);
#else
PyMODINIT_FUNC PyInit_gjk(void);
#endif

#endif /* !__PYX_HAVE__gjk */
