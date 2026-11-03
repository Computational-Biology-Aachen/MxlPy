/* Replaces SciPy's header of the same name, so lsoda.c and lsoda.h compile unchanged
 * without BLAS or LAPACK. The routines come from mxs_lapack.c, under an mxs_ prefix
 * so they never clash with a real BLAS loaded in the same process. */
#ifndef BLASLAPACK_DECLARATIONS_H
#define BLASLAPACK_DECLARATIONS_H

#define CBLAS_INT int
#define BLAS_FUNC(name) mxs_##name

void mxs_dgetrf(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
void mxs_dgetrs(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
void mxs_dgbtrf(int *m, int *n, int *kl, int *ku, double *ab, int *ldab, int *ipiv, int *info);
void mxs_dgbtrs(char *trans, int *n, int *kl, int *ku, int *nrhs, double *ab, int *ldab, int *ipiv, double *b,
                int *ldb, int *info);

#endif
