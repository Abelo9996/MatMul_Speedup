#include <immintrin.h>
#include <string.h>
#include <assert.h>
 
const char* dgemm_desc = "Simple blocked dgemm.";
 
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 192
#endif
 
#define min(a, b) (((a) < (b)) ? (a) : (b))
 
static double* to_aligned(int lda, int new_lda, double *A) {
   double *ret = _mm_malloc(new_lda * new_lda * sizeof(double), 64);
   for(int i = 0; i < new_lda; ++i) {
       memcpy(ret + i * new_lda, A + i * lda, sizeof(double) * lda);
       memset(ret + i * new_lda + lda, 0, sizeof(double) * (new_lda - lda));
   }
   return ret;
}
 
static void to_packed(int lda, int new_lda, double *A, double *B) {
   for(int i = 0; i < lda; ++i)
       memcpy(A + i * lda, B + i * new_lda, sizeof(double) * lda);
   _mm_free(B);
}
 
/*
* This auxiliary subroutine performs a smaller dgemm operation
*  C := C + A * B
* where C is M-by-N, A is M-by-K, and B is K-by-N.
*/
static void do_block(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C) {
   assert(M % 8 == 0 && N % 8 == 0 && K % 8 == 0);
   // For each row i of A
   for (int j = 0; j < N; j += 8){
       // For each column j of B
       for (int k = 0; k < K; k += 8) {
           double *b0 = B+j*lda+k, *b1 = b0+lda, *b2 = b1+lda, *b3 = b2+lda,
                  *b4 = b3+lda, *b5 = b4+lda, *b6 = b5+lda, *b7 = b6+lda;
           __m512d c0, c1, c2, c3, c4, c5, c6, c7;
           __m512d a0, a1, a2, a3, a4, a5, a6, a7;
           for (int i = 0; i < M; i += 8) {
               c0 = _mm512_load_pd(C + i + j * lda);
               c1 = _mm512_load_pd(C + i + j * lda + lda);
               c2 = _mm512_load_pd(C + i + j * lda + lda*2);
               c3 = _mm512_load_pd(C + i + j * lda + lda*3);
               c4 = _mm512_load_pd(C + i + j * lda + lda*4);
               c5 = _mm512_load_pd(C + i + j * lda + lda*5);
               c6 = _mm512_load_pd(C + i + j * lda + lda*6);
               c7 = _mm512_load_pd(C + i + j * lda + lda*7);
 
               a0 = _mm512_load_pd(A + i + k * lda);
               a1 = _mm512_load_pd(A + i + k * lda + lda);
               a2 = _mm512_load_pd(A + i + k * lda + lda*2);
               a3 = _mm512_load_pd(A + i + k * lda + lda*3);
               a4 = _mm512_load_pd(A + i + k * lda + lda*4);
               a5 = _mm512_load_pd(A + i + k * lda + lda*5);
               a6 = _mm512_load_pd(A + i + k * lda + lda*6);
               a7 = _mm512_load_pd(A + i + k * lda + lda*7);
              
/*[[[cog
import cog
for kk in range(8):
   for jj in range(8):
       cog.outl(f"c{jj} = _mm512_fmadd_pd(a{kk}, _mm512_set1_pd(b{jj}[{kk}]), c{jj});")
]]]*/
c0 = _mm512_fmadd_pd(a0, _mm512_set1_pd(b0[0]), c0);
c1 = _mm512_fmadd_pd(a0, _mm512_set1_pd(b1[0]), c1);
c2 = _mm512_fmadd_pd(a0, _mm512_set1_pd(b2[0]), c2);
c3 = _mm512_fmadd_pd(a0, _mm512_set1_pd(b3[0]), c3);
c4 = _mm512_fmadd_pd(a0, _mm512_set1_pd(b4[0]), c4);
c5 = _mm512_fmadd_pd(a0, _mm512_set1_pd(b5[0]), c5);
c6 = _mm512_fmadd_pd(a0, _mm512_set1_pd(b6[0]), c6);
c7 = _mm512_fmadd_pd(a0, _mm512_set1_pd(b7[0]), c7);
c0 = _mm512_fmadd_pd(a1, _mm512_set1_pd(b0[1]), c0);
c1 = _mm512_fmadd_pd(a1, _mm512_set1_pd(b1[1]), c1);
c2 = _mm512_fmadd_pd(a1, _mm512_set1_pd(b2[1]), c2);
c3 = _mm512_fmadd_pd(a1, _mm512_set1_pd(b3[1]), c3);
c4 = _mm512_fmadd_pd(a1, _mm512_set1_pd(b4[1]), c4);
c5 = _mm512_fmadd_pd(a1, _mm512_set1_pd(b5[1]), c5);
c6 = _mm512_fmadd_pd(a1, _mm512_set1_pd(b6[1]), c6);
c7 = _mm512_fmadd_pd(a1, _mm512_set1_pd(b7[1]), c7);
c0 = _mm512_fmadd_pd(a2, _mm512_set1_pd(b0[2]), c0);
c1 = _mm512_fmadd_pd(a2, _mm512_set1_pd(b1[2]), c1);
c2 = _mm512_fmadd_pd(a2, _mm512_set1_pd(b2[2]), c2);
c3 = _mm512_fmadd_pd(a2, _mm512_set1_pd(b3[2]), c3);
c4 = _mm512_fmadd_pd(a2, _mm512_set1_pd(b4[2]), c4);
c5 = _mm512_fmadd_pd(a2, _mm512_set1_pd(b5[2]), c5);
c6 = _mm512_fmadd_pd(a2, _mm512_set1_pd(b6[2]), c6);
c7 = _mm512_fmadd_pd(a2, _mm512_set1_pd(b7[2]), c7);
c0 = _mm512_fmadd_pd(a3, _mm512_set1_pd(b0[3]), c0);
c1 = _mm512_fmadd_pd(a3, _mm512_set1_pd(b1[3]), c1);
c2 = _mm512_fmadd_pd(a3, _mm512_set1_pd(b2[3]), c2);
c3 = _mm512_fmadd_pd(a3, _mm512_set1_pd(b3[3]), c3);
c4 = _mm512_fmadd_pd(a3, _mm512_set1_pd(b4[3]), c4);
c5 = _mm512_fmadd_pd(a3, _mm512_set1_pd(b5[3]), c5);
c6 = _mm512_fmadd_pd(a3, _mm512_set1_pd(b6[3]), c6);
c7 = _mm512_fmadd_pd(a3, _mm512_set1_pd(b7[3]), c7);
c0 = _mm512_fmadd_pd(a4, _mm512_set1_pd(b0[4]), c0);
c1 = _mm512_fmadd_pd(a4, _mm512_set1_pd(b1[4]), c1);
c2 = _mm512_fmadd_pd(a4, _mm512_set1_pd(b2[4]), c2);
c3 = _mm512_fmadd_pd(a4, _mm512_set1_pd(b3[4]), c3);
c4 = _mm512_fmadd_pd(a4, _mm512_set1_pd(b4[4]), c4);
c5 = _mm512_fmadd_pd(a4, _mm512_set1_pd(b5[4]), c5);
c6 = _mm512_fmadd_pd(a4, _mm512_set1_pd(b6[4]), c6);
c7 = _mm512_fmadd_pd(a4, _mm512_set1_pd(b7[4]), c7);
c0 = _mm512_fmadd_pd(a5, _mm512_set1_pd(b0[5]), c0);
c1 = _mm512_fmadd_pd(a5, _mm512_set1_pd(b1[5]), c1);
c2 = _mm512_fmadd_pd(a5, _mm512_set1_pd(b2[5]), c2);
c3 = _mm512_fmadd_pd(a5, _mm512_set1_pd(b3[5]), c3);
c4 = _mm512_fmadd_pd(a5, _mm512_set1_pd(b4[5]), c4);
c5 = _mm512_fmadd_pd(a5, _mm512_set1_pd(b5[5]), c5);
c6 = _mm512_fmadd_pd(a5, _mm512_set1_pd(b6[5]), c6);
c7 = _mm512_fmadd_pd(a5, _mm512_set1_pd(b7[5]), c7);
c0 = _mm512_fmadd_pd(a6, _mm512_set1_pd(b0[6]), c0);
c1 = _mm512_fmadd_pd(a6, _mm512_set1_pd(b1[6]), c1);
c2 = _mm512_fmadd_pd(a6, _mm512_set1_pd(b2[6]), c2);
c3 = _mm512_fmadd_pd(a6, _mm512_set1_pd(b3[6]), c3);
c4 = _mm512_fmadd_pd(a6, _mm512_set1_pd(b4[6]), c4);
c5 = _mm512_fmadd_pd(a6, _mm512_set1_pd(b5[6]), c5);
c6 = _mm512_fmadd_pd(a6, _mm512_set1_pd(b6[6]), c6);
c7 = _mm512_fmadd_pd(a6, _mm512_set1_pd(b7[6]), c7);
c0 = _mm512_fmadd_pd(a7, _mm512_set1_pd(b0[7]), c0);
c1 = _mm512_fmadd_pd(a7, _mm512_set1_pd(b1[7]), c1);
c2 = _mm512_fmadd_pd(a7, _mm512_set1_pd(b2[7]), c2);
c3 = _mm512_fmadd_pd(a7, _mm512_set1_pd(b3[7]), c3);
c4 = _mm512_fmadd_pd(a7, _mm512_set1_pd(b4[7]), c4);
c5 = _mm512_fmadd_pd(a7, _mm512_set1_pd(b5[7]), c5);
c6 = _mm512_fmadd_pd(a7, _mm512_set1_pd(b6[7]), c6);
c7 = _mm512_fmadd_pd(a7, _mm512_set1_pd(b7[7]), c7);
//[[[end]]]
 
               _mm512_store_pd(C + i + j * lda, c0);
               _mm512_store_pd(C + i + j * lda + lda * 1, c1);
               _mm512_store_pd(C + i + j * lda + lda * 2, c2);
               _mm512_store_pd(C + i + j * lda + lda * 3, c3);
               _mm512_store_pd(C + i + j * lda + lda * 4, c4);
               _mm512_store_pd(C + i + j * lda + lda * 5, c5);
               _mm512_store_pd(C + i + j * lda + lda * 6, c6);
               _mm512_store_pd(C + i + j * lda + lda * 7, c7);
           }
       }
   }
}
 
/* This routine performs a dgemm operation
*  C := C + A * B
* where A, B, and C are lda-by-lda matrices stored in column-major format.
* On exit, A and B maintain their input values. */
void square_dgemm_aligned(int lda, double* restrict A, double* restrict B, double* restrict C) {
   // For each block-row of A
   for (int k = 0; k < lda; k += BLOCK_SIZE) {
       // For each block-column of B
       for (int j = 0; j < lda; j += BLOCK_SIZE) {
           // Accumulate block dgemms into block of C
           for (int i = 0; i < lda; i += BLOCK_SIZE) {
               // Correct block dimensions if block "goes off edge of" the matrix
               int M = min(BLOCK_SIZE, lda - i);
               int N = min(BLOCK_SIZE, lda - j);
               int K = min(BLOCK_SIZE, lda - k);
               // Perform individual block dgemm
               do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
           }
       }
   }
}
 
void square_dgemm(int lda, double* A, double* B, double* restrict C) {
   int new_lda = lda + (lda % 8 ? 8-(lda%8) : 0);
   double *na = to_aligned(lda, new_lda, A), *nb = to_aligned(lda, new_lda, B), *nc = to_aligned(lda, new_lda, C);
   square_dgemm_aligned(new_lda, na, nb, nc);
   to_packed(lda, new_lda, C, nc);
}
