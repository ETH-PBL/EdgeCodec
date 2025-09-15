// #include "pmsis.h"
#include "stdio.h"
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include "vq_block.h"



/* Constant Variables */
// #define PERFORMANCE /* Print the Cycles and Active Cycles */
// #define VERBOSE  /* Print the intermediate results */

/* Functions */

/* Matrix Multiplication with pointers*/
// C = A*B where A is MxK, B is KxN, C is MxN
void matMul(float32_t *A, float32_t *B, float32_t *C, int rowSizeA, int colSizeA, int colSizeB)
{
    /*
        M is rowSize of A
        K is colSize of A and rowSize of B
        N is colSize of B
    */
    int i = 0; // index of row in A
    int j = 0; // index of col in B
    int k = 0; // index of col in A

    float32_t sum;
    for (i = 0; i < rowSizeA; i++) // row of A
    {   
        for (j = 0; j < colSizeB; j++) // col of B
        {
            sum = 0;
            for (k = 0; k < colSizeA; k++) // col of A
            {
                sum += A[i * colSizeA + k] * B[k * colSizeB + j];
            }
            C[i * colSizeB + j] = sum * -2;
        }
    }
}



void MatToVectorSum(float32_t *Src, float32_t *Dst,int rowSize, int colSize)
{   
/* 
    @brief: Perform Summation in the column dimension of the Input Matric
    @python_high_level: einops.reduce(Src, 'i d -> i', sum)
    @python_low_level: torch.sum(Src, axis=1) 
    @author: @MiirHo3eIN
    @param: float32_t *Src: Pointer to the Input Matrix
    @param: float32_t *Dst: Pointer to the Output Vector
    @param: int rowSize: Number of rows in the Input Matrix
    @param: int colSize: Number of columns in the Input Matrix

*/

    /* Sum the elements of each row */
    for (int i = 0; i < rowSize; i++)
    {
        for (int j = 0; j < colSize; j++)
        {
           
            Dst[i] += (Src[i*colSize + j]*Src[i*colSize + j] );
        }

        // printf("Sum of %d row is %f\n", i, Dst[i]);
    }
}


void VectorToMatrixAdd(float32_t *pSrcA, float32_t* pSrcB, float32_t *pDst, int rowSizeSrcA, int rowSizeSrcB)
{

/* 
    @brief: Perform Addition of two vectors and store the result in a matrix
    @author: @MiirHo3eIN
    @date: 2024-03-25
    @python: einops.rearrange(Src, 'i -> i 1') + einops.rearrange(Src, 'j -> 1 j')
    
    @param: float32_t *pSrcA: Pointer to the first Input Vector
    @param: float32_t *pSrcB: Pointer to the second Input Vector
    @param: float32_t *pDst: Pointer to the Output Matrix
    @param: int rowSizeSrcA: Number of rows in the first Input Vector
    @param: int rowSizeSrcB: Number of rows in the second Input Vector
*/

    int i;
    int j;
    
    for (i = 0; i < rowSizeSrcA; i++)
    {
        for (j = 0; j < rowSizeSrcB; j++)
        {
            pDst[i*rowSizeSrcA + j] = pSrcA[i] + pSrcB[j];
        }
    }
}

void MatrixAdd(float32_t *SrcA, float32_t *SrcB, float32_t *Dst, int rowSize, int colSize)
{

/* 
    @brief: Perform Addition of two matrices and store the result in a matrix. 
    @note: The two matrices must have the same dimensions [A1, A2] + [B1, B2] = [A1, A2] = [B1, B2]
    @author: @MiirHo3eIN
    @date: 2024-03-25
    @python: SrcA + SrcB

    @param: float32_t *SrcA: Pointer to the first Input Matrix
    @param: float32_t *SrcB: Pointer to the second Input Matrix
    @param: float32_t *Dst: Pointer to the Output Matrix
    @param: int rowSize: Number of rows in the Input Matrix
    @param: int colSize: Number of columns in the Input Matrix

*/
    for (int i = 0; i < rowSize; i++)
    {
        for (int j = 0; j < colSize; j++)
        {
            Dst[i * colSize + j] = Sqrt(SrcA[i * colSize + j] + SrcB[i * colSize + j]);
        }
    }
}



float32_t cdist(float32_t *SrcA, float32_t *SrcB, int rowSizeSrc, int colSizeSrc, bool verbose)
{

    /*
        @brief: Compute the Euclidean distance between two matrices.
        @note: The two matrices must have the same dimensions [A1, A2] + [B1, B2] = [A1, A2] = [B1, B2]
        @note: We compute A.B^T and then sum the elements of each row of A and B.
        @note: Having the same dimensions is crucial for the matrix addition operations followed by the matrix multiplication. 
        @note: The Euclidean distance is computed as follows: sqrt(-2xy + x + y)

        @author: @MiirHo3eIN
        @date: 2024-03-25
        @python: look at <vq_inference.py scripts for the two implementations of this function>
        @param: float32_t *SrcA: Pointer to the first Input Matrix
        @param: float32_t *SrcB: Pointer to the second Input Matrix
        @param: float32_t *Dst: Pointer to the Output Matrix
        @param: int rowSizeSrc: Number of rows in the Input Matrix
        @param: int colSizeSrc: Number of columns in the Input Matrix
    
    */
    
    float32_t *xTemp;
    float32_t *yTemp;
    float32_t *SrcBTransposed;
    float32_t *sumResult; 
    float32_t *Dst;

    uint32_t cycles = 0, tim_cycles  = 0;


    sumResult = (float32_t *) malloc(rowSizeSrc * rowSizeSrc * sizeof(float32_t));
    SrcBTransposed = (float32_t *) malloc(rowSizeSrc * colSizeSrc * sizeof(float32_t));
    xTemp = (float32_t *) malloc(rowSizeSrc * sizeof(float32_t));  
    yTemp = (float32_t *) malloc(rowSizeSrc * sizeof(float32_t)); // colSizeSrcA must be same as the rowSizeSrcB
    Dst = (float32_t *) malloc(rowSizeSrc * rowSizeSrc * sizeof(float32_t)); 

    /* Initialize xtemp and ytemp */
    VectorInit(xTemp, rowSizeSrc, 0, OFF);
    VectorInit(yTemp, rowSizeSrc, 0, OFF);
    MatrixInit(Dst, rowSizeSrc, rowSizeSrc);
    /* Print The two Input Matrices */
    #if defined (VERBOSE ) 
        MatrixPrint(SrcA, "SrcA", rowSizeSrc, colSizeSrc);
        MatrixPrint(SrcB, "SrcB", rowSizeSrc, colSizeSrc);
    #endif
    /* Embed Transpose */
    
    #if defined (PERFORMANCE )

        pi_perf_reset(); 
        pi_perf_start();
    #endif
    MatrixTranspose(SrcB, SrcBTransposed, rowSizeSrc, colSizeSrc); /* y = transpose(B) */
        
     #if defined (VERBOSE )     
        MatrixPrint(SrcBTransposed, "SrcBTransposed", colSizeSrc, rowSizeSrc); 
     #endif

    #if defined (PERFORMANCE )
        pi_perf_stop();
        cycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
        tim_cycles = pi_perf_read(PI_PERF_CYCLES);
        printf("Performance of the Transpose func : %d cycles Timer : %d cycles\n", cycles, tim_cycles);   
    #endif
   
   
   
    /* Matrix Multiplications */
        #if defined (PERFORMANCE )
        pi_perf_conf(1 << PI_PERF_CYCLES | 1 << PI_PERF_ACTIVE_CYCLES);     
        pi_perf_reset(); 
        pi_perf_start();
        #endif
    matMul(SrcA, SrcBTransposed, Dst, rowSizeSrc, colSizeSrc, rowSizeSrc); /* -2xy */
        #if defined (PERFORMANCE )
        pi_perf_stop();
        cycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
        tim_cycles = pi_perf_read(PI_PERF_CYCLES);
        printf("Performance of the MatMul func : %d cycles Timer : %d cycles\n", cycles, tim_cycles);
        #endif

    free(SrcBTransposed);
    #if defined (VERBOSE ) 
         MatrixPrint(Dst, "MatMul Result", rowSizeSrc, rowSizeSrc);
    #endif
    /* Sum the elements of each row */
    #if defined (PERFORMANCE )
        pi_perf_reset(); 
        pi_perf_start();
    #endif

    MatToVectorSum(SrcA, xTemp, rowSizeSrc, colSizeSrc); /* reduce(x, 'i d -> i', sum) */
    MatToVectorSum(SrcB, yTemp, rowSizeSrc, colSizeSrc); /* reduce(y, 'i d -> i', sum) */
    #if defined (PERFORMANCE )
        pi_perf_stop();
        cycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
        tim_cycles = pi_perf_read(PI_PERF_CYCLES);
        printf("Performance of the two MatToVector func : %d cycles Timer : %d cycles\n", cycles, tim_cycles);
    #endif
        #if defined (VERBOSE ) 
        VectorPrint(xTemp, "xTemp", rowSizeSrc);
        VectorPrint(yTemp, "yTemp", rowSizeSrc);
        #endif 
    
    /* Add x and y*/
        #if defined (PERFORMANCE )
        pi_perf_reset(); 
        pi_perf_start();
        #endif
    VectorToMatrixAdd(xTemp, yTemp, sumResult, rowSizeSrc, rowSizeSrc); /* x + y */
        #if defined (PERFORMANCE )
        pi_perf_stop();
        cycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
        tim_cycles = pi_perf_read(PI_PERF_CYCLES);
        printf("Performance of the VectorToMatAdd func : %d cycles Timer : %d cycles\n", cycles, tim_cycles);
        #endif

    free(xTemp);
    free(yTemp);
    
    #if defined (VERBOSE )     
    MatrixPrint(sumResult, "SumResult", rowSizeSrc, rowSizeSrc);    
    #endif
    /* Add the result of matrix multiplication and sum of x and y */
        #if defined (PERFORMANCE )
        pi_perf_reset(); 
        pi_perf_start();
        #endif
    MatrixAdd(Dst, sumResult, Dst, rowSizeSrc, rowSizeSrc); /* -2xy + x + y */
        #if defined (PERFORMANCE )
        pi_perf_stop();
        cycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
        tim_cycles = pi_perf_read(PI_PERF_CYCLES);
        printf("Performance of the MatrixAdd_W_SQRT func : %d cycles Timer : %d cycles\n", cycles, tim_cycles);
        #endif

    #if defined (VERBOSE )     
    MatrixPrint(Dst, "Result", rowSizeSrc, rowSizeSrc); /* -2xy + x + y */
    #endif
    
    return Dst[0]; /* Return the first element of the result matrix */
}
