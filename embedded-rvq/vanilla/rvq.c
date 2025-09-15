/* PMSIS includes */
//#include "pmsis.h" // <--------- this was used for GAP9
#include "stdio.h"
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#include "vq_block.h"
#include "GVSOC_output.h" //<--------- input file provided
#include "float16_codebooks.h" // <--------- codebook provided


/* User Defines */
#define  N_ROW 1
#define  N_COL 100
#define  N_CHANNELS 9
#define CODEBOOK_DIM 768
#define NUM_QUANTIZERS 4

#define PROFILE 0

// algorithm proposed by SoundStream: https://arxiv.org/pdf/2107.03312
//
//  Input: y = enc(x) the output of the encoder, quantizers Q_i for i = 1 .. N_q
//  Output: quanted y_dash
//
//  y_dash = 0.0
//  residual = y
//  for i = 1 to N_q do:
//      y_dash += Q_i(residual)
//      residual -= Q_i(residual)
//  return y_dash

// /* Program Entry. */
int main(void)
{
    printf("\n\n\t *** RVQ by Beni ***\n\n");

    // Allocate Memory
    float32_t *residual = (float32_t *)malloc(N_CHANNELS * N_COL * sizeof(float32_t));
    float32_t *codeword_ids = (float32_t *)malloc(N_CHANNELS * NUM_QUANTIZERS * sizeof(float32_t));
    // probably in final run, change the codewords to int32_t, currently using float so I can print them
    if (!residual || !codeword_ids)
    {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize arrays
    VectorInit((float32_t *)codeword_ids, N_CHANNELS * NUM_QUANTIZERS, -1, OFF);

    printf("Starting Residual Quantization...\n");
    #if defined (PROFILE)
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    // Iterate over channels
    for (int i = 0; i < N_CHANNELS; i++)
    {
        // Initialize residual with the input
        for (int j = 0; j < N_COL; j++)
        {
            residual[i * N_COL + j] = input[i][j]; // residual is a "flattened" 2d array, so I have to multiply with index
        }

        // Iterate over quantizers (codebooks)
        for (int q = 0; q < NUM_QUANTIZERS; q++)
        {
            float32_t cdist_current = 1e9; // Large initial value for minimum distance
            int16_t best_codeword = -1; // allocate this in memory and pass a pointer/reference to the core functions to change this

            // Search for the best matching codeword in the current codebook
            for (int k = 0; k < CODEBOOK_DIM; k++)
            {
                float32_t cdist_next = cdist(
                    &residual[i * N_COL],
                    &codebooks[q][k * N_COL],
                    N_ROW, N_COL, OFF);
                if (cdist_next < cdist_current) //critical-----------
                {
                    cdist_current = cdist_next; 
                    best_codeword = k;
                } //------------------------------
            }

            // Save the selected codeword ID
            codeword_ids[i * NUM_QUANTIZERS + q] = best_codeword;

            // Update residual by subtracting the contribution of the best codeword
            for (int j = 0; j < N_COL; j++)
            {
                float32_t quantized_value = codebooks[q][best_codeword * N_COL + j];
                residual[i * N_COL + j] -= quantized_value;
            }
        }
    }

    #if defined (PROFILE)
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time: %f seconds\n", elapsed);
    #endif
    // Print Codeword Indexes
    MatrixPrint(codeword_ids, "", N_CHANNELS, NUM_QUANTIZERS);

    // Free Memory
    free(residual);
    free(codeword_ids);

    printf("Done!\n");
    return 0;
}



