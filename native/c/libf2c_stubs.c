/*
 * libf2c_stubs.c — minimal stubs for libf2c routines used by radau5.c
 *
 * RADAU5 only calls these on error paths (WRITE(6,*) messages) or for
 * arithmetic helpers. Emscripten has no Fortran I/O, so I/O stubs are
 * silent. Math helpers are thin wrappers around <math.h>.
 */

#include "f2c.h"

/* ------------------------------------------------------------------ */
/* Fortran I/O stubs — silently dropped                                */
/* ------------------------------------------------------------------ */

int s_wsle(cilist *a) { (void)a; return 0; }
int e_wsle(void)      { return 0; }
int do_lio(integer *type, integer *number, char *ptr, ftnlen len) {
    (void)type; (void)number; (void)ptr; (void)len;
    return 0;
}

int s_wsfe(cilist *a) { (void)a; return 0; }
int e_wsfe(void)      { return 0; }
int do_fio(integer *type, char *ptr, ftnlen len) {
    (void)type; (void)ptr; (void)len;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Arithmetic helpers                                                  */
/* ------------------------------------------------------------------ */

doublereal d_sign(doublereal *a, doublereal *b) {
    doublereal x = (*a >= 0.0) ? *a : -*a;
    return (*b >= 0.0) ? x : -x;
}

doublereal pow_dd(doublereal *a, doublereal *b) {
    return pow(*a, *b);
}

doublereal pow_di(doublereal *a, integer *n) {
    doublereal r = 1.0;
    integer    i = *n;
    doublereal x = *a;
    if (i < 0) { x = 1.0 / x; i = -i; }
    for (; i > 0; i--) r *= x;
    return r;
}
