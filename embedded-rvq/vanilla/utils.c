//#include "pmsis.h"
#include "stdio.h"
#include <stdint.h> 
#include <math.h>

#include "vq_block.h"





void MatrixInit(float32_t *pSrc, int rowSize, int colSize)
{

/*

    @brief: Initialize the Matrix with incremental values from 1 to rowSize*colSize in row major order.
    @author: @MiirHo3eIN
    @date: 2024-03-25

    @param: float32_t *pSrc: Pointer to the Matrix
    @param: int rowSize: Number of rows in the Matrix
    @param: int colSize: Number of columns in the Matrix

*/

    int i;
    int j;
    for (i = 0; i < rowSize; i++)
    {
        for (j = 0; j < colSize; j++)
        {
            pSrc[i * colSize + j] = (i*colSize) +1 + j; // A[i][j] = i*colSize + j + 1
        }
    }
}


void MatrixPrint(float *A, char *matrix_name, int rowSize, int colSize)
{

/*
    @brief: Print the Matrix elements with the name of the Matrix provided in the argument matrix_name
    @author: @MiirHo3eIN
    @date: 2024-03-25
    
    @param: float *A: Pointer to the Matrix
    @param: char *matrix_name: Name of the Matrix
    @param: int rowSize: Number of rows in the Matrix
    @param: int colSize: Number of columns in the Matrix

*/


    printf("%s=", matrix_name); 
    for (int i = 0; i < rowSize; i++) 
    {   printf("[");
        for (int j = 0; j < colSize; j++) 
        {
            printf("%f, ", A[i*colSize + j]);
            // printf("]");
        }
        printf("];\n");

    }

}



void VectorInit(float32_t *pSrc, int rowSize, float32_t value, bool is_incremental)
{

/* 
    @brief: Initialize the Vector with a specific value or incremental values from 1 to rowSize
    @author: @MiirHo3eIN
    @date: 2024-03-25

    @param: float32_t *pSrc: Pointer to the Vector
    @param: int rowSize: Number of elements in the Vector
    @param: float32_t value: Value to be initialized in the Vector
    @param: bool is_incremental: If ON, 
    the vector will be initialized with incremental values from 1 to rowSize, 
    otherwise, the vector will be initialized with the value provided in the argument
    value.

*/


    int i; 

    if (is_incremental == ON)
    {
        for (i = 0; i < rowSize; i++)
        {
            pSrc[i] = (i+1);
        }
    }
    else
        for (i = 0; i < rowSize; i++)
    {
        pSrc[i] = 1 * value;
    }
}




void VectorPrint(float32_t *pSrc, char *name , int rowSize)
{
/*
    @brief: Print the Vector elements with the name of the Vector provided in the argument name
    @author: @MiirHo3eIN
    @date: 2024-03-25

    @param: float32_t *pSrc: Pointer to the Vector
    @param: char *name: Name of the Vector
    @param: int rowSize: Number of elements in the Vector

*/

    int i;
    printf("Vector %s = \n", name);
    for (i = 0; i < rowSize; i++)
    {   
        printf("%f\n", pSrc[i]);
    }
}



void MatrixTranspose(float32_t *pSrc, float32_t *pDst, int rowSize, int colSize)
{

/*
    @brief: Perform Transpose of the Input Matrix and store the result in the Output Matrix
    @author: @MiirHo3eIN
    @date: 2024-03-25

    @param: float32_t *Src: Pointer to the Input Matrix
    @param: float32_t *Dst: Pointer to the Output Matrix
    @param: int rowSize: Number of rows in the Input Matrix
    @param: int colSize: Number of columns in the Input Matrix

*/


    for (int i = 0; i < rowSize; i++)
    {
        for (int j = 0; j < colSize; j++)
        {
            pDst[j * rowSize + i] = pSrc[i * colSize + j];
        }
    }
}


/*
    @brief: Sqrt function implemented using assembly instruction fsqrt.s
    @author: @ah-kiamarzi
    
    @param: float x: Input value
    @return: float: Square root of the input value
*/
 float Sqrt(float x) {
        
        return sqrtf(x);
}


