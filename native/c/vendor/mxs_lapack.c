/* The four LAPACK routines lsoda.c calls, unblocked and column-major, following
 * LAPACK's dgetf2/dgetrs/dgbtf2/dgbtrs. Pivots are 1-based like LAPACK's. Only
 * trans = "N" is supported, which is all lsoda.c uses. Results match LAPACK's up to
 * rounding, since a blocked LAPACK orders its operations differently. */
#include <math.h>
#include <stddef.h>
#include "blaslapack_declarations.h"

static int imin(int a, int b) { return a < b ? a : b; }
static int imax(int a, int b) { return a > b ? a : b; }

/* LU factorisation with partial pivoting of the m-by-n matrix a. */
void mxs_dgetrf(int *m, int *n, double *a, int *lda, int *ipiv, int *info) {
    const int M = *m, N = *n, L = *lda;
#define A(i, j) a[(i) + (size_t)(j) * L]
    *info = 0;
    for (int j = 0; j < imin(M, N); j++) {
        int p = j;
        for (int i = j + 1; i < M; i++) {
            if (fabs(A(i, j)) > fabs(A(p, j))) p = i;
        }
        ipiv[j] = p + 1;
        if (A(p, j) == 0.0) {
            if (*info == 0) *info = j + 1;
            continue;
        }
        if (p != j) {
            for (int k = 0; k < N; k++) {
                double t = A(j, k);
                A(j, k) = A(p, k);
                A(p, k) = t;
            }
        }
        const double r = 1.0 / A(j, j);
        for (int i = j + 1; i < M; i++) A(i, j) *= r;
        for (int k = j + 1; k < N; k++) {
            const double s = A(j, k);
            if (s == 0.0) continue;
            for (int i = j + 1; i < M; i++) A(i, k) -= A(i, j) * s;
        }
    }
#undef A
}

/* Solves A X = B with the factors from mxs_dgetrf. */
void mxs_dgetrs(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info) {
    const int N = *n, L = *lda;
#define A(i, j) a[(i) + (size_t)(j) * L]
    *info = (*trans == 'N' || *trans == 'n') ? 0 : -1;
    if (*info) return;
    for (int r = 0; r < *nrhs; r++) {
        double *x = b + (size_t)r * *ldb;
        for (int i = 0; i < N; i++) {
            const int p = ipiv[i] - 1;
            if (p != i) {
                double t = x[i];
                x[i] = x[p];
                x[p] = t;
            }
        }
        for (int j = 0; j < N; j++) {
            for (int i = j + 1; i < N; i++) x[i] -= A(i, j) * x[j];
        }
        for (int j = N - 1; j >= 0; j--) {
            x[j] /= A(j, j);
            for (int i = 0; i < j; i++) x[i] -= A(i, j) * x[j];
        }
    }
#undef A
}

/* LU factorisation of a band matrix with kl sub- and ku superdiagonals, stored in rows
 * kl..2*kl+ku of ab (0-based). The first kl rows receive the fill-in. */
void mxs_dgbtrf(int *m, int *n, int *kl, int *ku, double *ab, int *ldab, int *ipiv, int *info) {
    const int M = *m, N = *n, KL = *kl, KU = *ku, L = *ldab, KV = KU + KL;
    /* 1-based indices, as in LAPACK's dgbtf2. */
#define AB(i, j) ab[((i) - 1) + (size_t)((j) - 1) * L]
    *info = 0;
    for (int j = KU + 2; j <= imin(KV, N); j++) {
        for (int i = KV - j + 2; i <= KL; i++) AB(i, j) = 0.0;
    }
    int ju = 1;
    for (int j = 1; j <= imin(M, N); j++) {
        if (j + KV <= N) {
            for (int i = 1; i <= KL; i++) AB(i, j + KV) = 0.0;
        }
        const int km = imin(KL, M - j);
        int jp = 1;
        for (int i = 2; i <= km + 1; i++) {
            if (fabs(AB(KV + i, j)) > fabs(AB(KV + jp, j))) jp = i;
        }
        ipiv[j - 1] = jp + j - 1;
        if (AB(KV + jp, j) == 0.0) {
            if (*info == 0) *info = j;
            continue;
        }
        ju = imax(ju, imin(j + KU + jp - 1, N));
        if (jp != 1) {
            /* Swap rows jp+j-1 and j across columns j..ju. */
            for (int k = 0; k <= ju - j; k++) {
                double *p = &AB(KV + jp - k, j + k), *q = &AB(KV + 1 - k, j + k);
                double t = *p;
                *p = *q;
                *q = t;
            }
        }
        if (km > 0) {
            const double r = 1.0 / AB(KV + 1, j);
            for (int i = 1; i <= km; i++) AB(KV + 1 + i, j) *= r;
            for (int k = 1; k <= ju - j; k++) {
                const double s = AB(KV + 1 - k, j + k);
                if (s == 0.0) continue;
                for (int i = 1; i <= km; i++) AB(KV + 1 + i - k, j + k) -= AB(KV + 1 + i, j) * s;
            }
        }
    }
#undef AB
}

/* Solves A X = B with the factors from mxs_dgbtrf. */
void mxs_dgbtrs(char *trans, int *n, int *kl, int *ku, int *nrhs, double *ab, int *ldab, int *ipiv, double *b,
                int *ldb, int *info) {
    const int N = *n, KL = *kl, KU = *ku, L = *ldab, KD = KU + KL + 1;
#define AB(i, j) ab[((i) - 1) + (size_t)((j) - 1) * L]
    *info = (*trans == 'N' || *trans == 'n') ? 0 : -1;
    if (*info) return;
    for (int r = 0; r < *nrhs; r++) {
        double *x = b + (size_t)r * *ldb - 1; /* x[1..N] */
        if (KL > 0) {
            for (int j = 1; j <= N - 1; j++) {
                const int lm = imin(KL, N - j), l = ipiv[j - 1];
                if (l != j) {
                    double t = x[l];
                    x[l] = x[j];
                    x[j] = t;
                }
                for (int i = 1; i <= lm; i++) x[j + i] -= AB(KD + i, j) * x[j];
            }
        }
        /* Upper triangular band solve with kl+ku superdiagonals (dtbsv). */
        const int K = KL + KU;
        for (int j = N; j >= 1; j--) {
            if (x[j] == 0.0) continue;
            x[j] /= AB(KD, j);
            const double t = x[j];
            for (int i = j - 1; i >= imax(1, j - K); i--) x[i] -= t * AB(KD + i - j, j);
        }
    }
#undef AB
}
