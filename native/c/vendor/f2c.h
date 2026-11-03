/* f2c.h — minimal subset needed for f2c-generated RADAU5/dc_decsol code */

#ifndef F2C_H
#define F2C_H

#include <math.h>
#include <stdlib.h>

typedef int     integer;
typedef double  doublereal;
typedef float   real;
typedef int     logical;
typedef int     ftnlen;

/* Function pointer types used by f2c for external/callback arguments */
typedef int (*S_fp)();    /* subroutine pointer (known args) */
typedef int (*U_fp)();    /* subroutine pointer (unknown args) */
typedef double (*D_fp)(); /* double function pointer */

/* Logical constants */
#define TRUE_  1
#define FALSE_ 0

/* min/max macros (f2c style) */
#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))
#undef  abs
#define abs(x)   ((x) >= 0 ? (x) : -(x))
#define dabs(x)  fabs(x)
#define dmin(a,b) (double)min(a,b)
#define dmax(a,b) (double)max(a,b)
#define dsqrt(x) sqrt(x)

/* Sequential I/O control list — matches f2c-generated cilist literals   *
 * { cierr, ciunit, ciend, cifmt, cirec }                                */
typedef struct {
    int   cierr;   /* 0 = no error checking */
    int   ciunit;  /* Fortran unit number    */
    int   ciend;   /* end-of-file handling   */
    char *cifmt;   /* format string (NULL for list-directed) */
    int   cirec;   /* record number (direct access)          */
} cilist;

/* I/O function declarations (all no-ops in libf2c_stubs.c) */
extern int s_wsle(cilist *);
extern int e_wsle(void);
extern int do_lio(integer *, integer *, char *, ftnlen);
extern int s_wsfe(cilist *);
extern int e_wsfe(void);
extern int do_fio(integer *, char *, ftnlen);

/* Math helpers (implemented in libf2c_stubs.c) */
extern doublereal d_sign(doublereal *a, doublereal *b);
extern doublereal pow_dd(doublereal *a, doublereal *b);
extern doublereal pow_di(doublereal *a, integer *n);

#endif /* F2C_H */
