#include <stdint.h>
#include <math.h>
// #include "pmsis.h"
#include "stdio.h"

typedef float float16_t;
typedef float float32_t;
typedef enum {ON, OFF} bool;
#define float32_t float16_t

/* Vq Kernels Functions */

void matMul(float32_t *A, float32_t *B, float32_t *C, int rowSizeA, int colSizeA, int colSizeB);
void MatToVectorSum(float32_t *Src, float32_t *Dst,int rowSize, int colSize);
void VectorToMatrixAdd(float32_t *pSrcA, float32_t* pSrcB, float32_t *pDst, int rowSizeSrcA, int rowSizeSrcB);
void MatToVectorSum(float32_t *Src, float32_t *Dst,int rowSize, int colSize);
void VectorToMatrixAdd(float32_t *pSrcA, float32_t* pSrcB, float32_t *pDst, int rowSizeSrcA, int rowSizeSrcB);
void MatrixAdd(float32_t *SrcA, float32_t *SrcB, float32_t *Dst, int rowSize, int colSize);
float32_t cdist(float32_t *SrcA, float32_t *SrcB, int rowSizeSrc, int colSizeSrc, bool verbose);


/* Utils Functions */
void MatrixInit(float32_t *pSrc, int rowSize, int colSize);
void MatrixPrint(float *A, char *matrix_name, int rowSize, int colSize);
void VectorInit(float32_t *pSrc, int rowSize, float32_t value, bool is_incremental);
void VectorPrint(float32_t *pSrc, char *vector_name, int rowSize);
void MatrixTranspose(float32_t *pSrc, float32_t *pDst, int rowSize, int colSize);
float Sqrt(float x); 