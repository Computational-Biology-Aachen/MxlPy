/* RADAU5 for MxlPy's Native integrator, returning values at requested time points.
 * The same source builds the native wheel and radau.wasm. Solver state is global,
 * so only one solve can run at a time in a process. */
#include "f2c.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define MXN_API __declspec(dllexport)
#else
#define MXN_API
#endif

extern int radau5_(integer *n, U_fp fcn, doublereal *x, doublereal *y, doublereal *xend, doublereal *h,
                   doublereal *rtol, doublereal *atol, integer *itol, U_fp jac, integer *ijac, integer *mljac,
                   integer *mujac, U_fp mas, integer *imas, integer *mlmas, integer *mumas, U_fp solout,
                   integer *iout, doublereal *work, integer *lwork, integer *iwork, integer *liwork,
                   doublereal *rpar, integer *ipar, integer *idid);
extern doublereal contr5_(integer *i, doublereal *x, doublereal *cont, integer *lrc);
extern double radau5_jac_growth;

typedef void (*ModelFn)(int, double, double *, double *, double *);
static ModelFn g_model;
static const double *g_t_out;
static double *g_y_out;
static int g_n_out, g_next;

/* A function pointer natively, a table index in wasm. */
MXN_API void mxn_set_model(intptr_t model) { g_model = (ModelFn)model; }

static int fcn_dispatch(integer *n, doublereal *x, doublereal *y, doublereal *f, doublereal *rpar, integer *ipar) {
    (void)ipar;
    g_model((int)*n, *x, y, f, rpar);
    return 0;
}

static int no_jac(void) { return 0; }
static int no_mas(void) { return 0; }

static int solout(integer *nr, doublereal *xold, doublereal *x, doublereal *y, doublereal *cont, integer *lrc,
                  integer *n, doublereal *rpar, integer *ipar, integer *irtrn) {
    (void)nr; (void)xold; (void)rpar; (void)ipar;
    *irtrn = 1;
    while (g_next < g_n_out && g_t_out[g_next] <= *x) {
        double *row = g_y_out + (size_t)g_next * (size_t)*n;
        doublereal s = g_t_out[g_next];
        if (s == *x) {
            memcpy(row, y, (size_t)*n * sizeof(double));
        } else {
            for (integer i = 1; i <= *n; i++) row[i - 1] = contr5_(&i, &s, cont, lrc);
        }
        g_next++;
    }
    return 0;
}

/* Integrates from t_out[0] to t_out[n_out - 1], writing rows of n values into y_out.
 * y holds the initial state on entry and the final state on return. Returns RADAU5's
 * IDID: positive on success, -1 bad input, -2 too many steps, -3 step too small,
 * -4 singular matrix. Returns -5 if allocation fails or the result isn't finite. */
MXN_API int mxn_radau(int n, double *y, double *pars, double rtol, double atol, int nmax, int reuse_jacobian,
                      int n_out, const double *t_out, double *y_out) {
    integer N = n, itol = 0, ijac = 0, mljac = n, mujac = n, imas = 0, mlmas = n, mumas = n, iout = 1, idid = 0, ipar = 0;
    integer lwork = 4 * N * N + 12 * N + 20, liwork = 3 * N + 20;
    doublereal *work = calloc((size_t)lwork, sizeof(doublereal));
    integer *iwork = calloc((size_t)liwork, sizeof(integer));
    if (!work || !iwork) {
        free(work);
        free(iwork);
        return -5;
    }
    doublereal x = t_out[0], xend = t_out[n_out - 1], h = 0.0, rt = rtol, at = atol;
    iwork[0] = 1;    /* IWORK(1): Hessenberg form */
    iwork[1] = nmax; /* IWORK(2): maximal number of steps */
    radau5_jac_growth = 0.0;
    if (reuse_jacobian) {
        /* Faster on larger stiff models, less accurate on some small ones, and too
         * rough for finite differences across parameters. See misc/mxl-perf. */
        iwork[2] = 10;
        work[2] = 0.05;
        work[5] = 1.5;
        radau5_jac_growth = 8.0;
    }
    g_t_out = t_out;
    g_y_out = y_out;
    g_n_out = n_out;
    g_next = 0;
    radau5_(&N, (U_fp)fcn_dispatch, &x, y, &xend, &h, &rt, &at, &itol, (U_fp)no_jac, &ijac, &mljac, &mujac,
            (U_fp)no_mas, &imas, &mlmas, &mumas, (U_fp)solout, &iout, work, &lwork, iwork, &liwork, pars, &ipar, &idid);
    free(work);
    free(iwork);
    if (idid > 0 && g_next != n_out) return -5;
    if (idid > 0) {
        for (int i = 0; i < n; i++) {
            if (!isfinite(y[i])) return -5;
        }
    }
    return (int)idid;
}

#ifdef MXN_WASM
/* Model math stays inside wasm, rather than calling back into Python. */
#define F1(name) double mxn_##name(double x) { return name(x); }
F1(exp) F1(log) F1(sin) F1(cos) F1(tan) F1(asin) F1(acos) F1(atan)
F1(sinh) F1(cosh) F1(tanh) F1(asinh) F1(acosh) F1(atanh)
double mxn_pow(double a, double b) { return pow(a, b); }
#else
/* Loaded with ctypes, but setuptools builds it as an extension and Windows linkers
 * insist on this symbol. It's never imported. */
MXN_API void PyInit__radau(void) {}
#endif
