/* PMSIS includes */
#include "pmsis.h"
#include "stdio.h"
#include <stdint.h>
#include <math.h>

#include "vq_block.h"
#include "input_0.h"
/* Scenarios to explore 

    1. 
*/

/* Variables used. */


#define N_ROW 1 // row size
#define N_COL 100 // column size

#define N_ROW_A 10
#define N_COL_A 20

#define N_ROW_B 20
#define N_COL_B 10




void MatrixTransposeTest() 
{
    float32_t *A;
    float32_t *B;
    A = (float32_t *) malloc(N_ROW * N_COL * sizeof(float32_t));
    B = (float32_t *) malloc(N_ROW * N_COL * sizeof(float32_t));

    MatrixInit(A, N_ROW, N_COL);
    MatrixInit(B, N_COL, N_ROW);

    MatrixPrint(A, "A", N_ROW, N_COL);
    MatrixTranspose(A, B, N_ROW, N_COL);
    MatrixPrint(B, "A Transpose", N_COL, N_ROW);
}

void VectorAddTest()
{
    /* Init the Vectors */
    float32_t *A;
    float32_t *B;
    float32_t *C;

    A = (float32_t *) malloc(N_ROW * sizeof(float32_t));
    B = (float32_t *) malloc(N_COL * sizeof(float32_t));
    C = (float32_t *) malloc(N_ROW * N_COL *sizeof(float32_t));

    VectorInit(A, N_ROW, 10, ON);
    VectorInit(B, N_COL, 5, ON);
    VectorInit(C, N_ROW*N_COL, 0, OFF);
    VectorToMatrixAdd(A, B, C, N_ROW, N_COL);

    VectorPrint(A, "A", N_ROW);
    VectorPrint(B, "B", N_COL);
    VectorPrint(C, "C", N_ROW*N_COL);

}


void MatrixAddTest()
{
    float32_t *A;
    float32_t *B;
    float32_t *C; 
    A = (float32_t *) malloc(N_COL*N_ROW * sizeof(float32_t));
    B = (float32_t *) malloc(N_COL*N_ROW * sizeof(float32_t));
    C = (float32_t *) malloc(N_ROW * N_ROW * sizeof(float32_t));

    
    // Initialize the matrices
    MatrixInit(A, N_ROW, N_COL);
    MatrixInit(B, N_ROW, N_COL);
    MatrixInit(C, N_ROW, N_ROW);

    // Print the matrices
    MatrixPrint(A, "A", N_ROW, N_COL);
    MatrixPrint(B, "B", N_ROW, N_COL);

    // Add the matrices
    MatrixAdd(A, B, C, N_ROW, N_COL);

    // Print the result
    MatrixPrint(C, "C", N_ROW, N_COL);
}


void cdist_test()
{
    float32_t *A;
    float32_t *B;
    float32_t *C; 
    A = (float32_t *) malloc(N_COL*N_ROW * sizeof(float32_t));
    B = (float32_t *) malloc(N_COL*N_ROW * sizeof(float32_t));
    C = (float32_t *) malloc(N_ROW * N_ROW * sizeof(float32_t));

    
    // Initialize the matrices
    MatrixInit(A, N_ROW, N_COL );
    MatrixInit(B, N_ROW, N_COL );
    MatrixInit(C, N_ROW, N_ROW );

    // Print the matrices
    cdist(A, B, C, N_ROW, N_COL);
}


void MatrixRead_test()
{
    float32_t *A;
    // A = (float32_t *) malloc(N_COL*N_ROW * sizeof(float32_t));
    MatrixPrint((float32_t *)input_0, "Sample_0", N_ROW, N_COL);
}

/* Program Entry. */
int main(void)
{
    printf("\n\n\t *** PMSIS HelloWorld ***\n\n");
    printf("Entering main controller\n");

    uint32_t errors = 0;
    
    // MatrixTransposeTest();
    // VectorAddTest();
    // MatrixAddTest();
    // cdist_test();
    MatrixRead_test();
    printf("Bye !\n");

    return errors;
}
