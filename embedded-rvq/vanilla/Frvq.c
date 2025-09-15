#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#include "vq_block.h"
#include "GVSOC_output.h"
#include "float16_codebooks.h"

/* User Defines */
#define N_ROW 1
#define N_COL 100 // This is latent space specific (N_COL and N_CHANNELS), if you decided to use a different model or rather latent space, then you need to change them accordingly.
#define N_CHANNELS 9
#define CODEBOOK_DIM 768 // CODEBOOK_DIM and NUM_QUANTIZERS are specific to the residual vector quantizer.
#define NUM_QUANTIZERS 4
#define N_THREADS 16  // Number of threads 


/*
This follows the rvq.c implementation, but uses threading to parallelize the search for the best codeword.
This is regular C code, not MCU specific!
*/

// #define PROFILE 0


typedef struct {
    float32_t dist;
    int codeword;
} Tuple;

typedef struct {
    int ID; 
    int start; 
    int end;
    int channel; 
    int quantizer; 
    float32_t *residual;       
    float32_t *codebook;        
    Tuple *results;       
} ThreadArgs;

/* Thread Function */
void *thread_func(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    int start = args->start;
    int end = args->end;
    int channel = args->channel;
    int quantizer = args->quantizer;
    float32_t *residual = args->residual;
    float32_t *codebook = args->codebook;
    Tuple *results = args->results;

    float32_t local_cdist = 1e9;
    int local_codeword = -1;


    for (int k = start; k < end; k++) {
        float32_t cdist_next = cdist(
            &residual[channel * N_COL],
            &codebook[k * N_COL],
            N_ROW, N_COL, OFF);
        if (cdist_next < local_cdist) {
            local_cdist = cdist_next;
            local_codeword = k;
        }
    }

    results[args->ID].dist = local_cdist;
    results[args->ID].codeword = local_codeword;

    return NULL;
}

/* Function to Find Best Codeword */
int find_best_codeword(Tuple *results, int num_threads) {
    float32_t best_dist = 1e9;
    int best_codeword = -1;

    for (int i = 0; i < num_threads; i++) {
        if (results[i].dist < best_dist) {
            best_dist = results[i].dist;
            best_codeword = results[i].codeword;
        }
    }

    return best_codeword;
}

int main(void) {
    printf("\n\n\t *** Threaded RVQ ***\n\n");

    // Allocate Memory
    float32_t *residual = (float32_t *)malloc(N_CHANNELS * N_COL * sizeof(float32_t));
    float32_t *codeword_ids = (float32_t *)malloc(N_CHANNELS * NUM_QUANTIZERS * sizeof(float32_t));
    Tuple *results = (Tuple *)malloc(N_THREADS * sizeof(Tuple));
    if (!residual || !codeword_ids || !results) {
        printf("Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize arrays
    VectorInit((float32_t *)codeword_ids, N_CHANNELS * NUM_QUANTIZERS, -1, OFF);
    printf("Starting Residual Quantization...\n");

    pthread_t threads[N_THREADS];
    ThreadArgs thread_args[N_THREADS];

    #if defined (PROFILE)
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    #endif

    // Iterate over channels
    for (int i = 0; i < N_CHANNELS; i++) {
        // Initialize residual with the input
        for (int j = 0; j < N_COL; j++) {
            residual[i * N_COL + j] = input[i][j];
        }

        // Iterate over quantizers (codebooks)
        for (int q = 0; q < NUM_QUANTIZERS; q++) {

            int chunk_size = CODEBOOK_DIM / N_THREADS; // BE CAREFUL, only works here because I know it will be an int
            for (int t = 0; t < N_THREADS; t++) {
                thread_args[t].ID = t;
                thread_args[t].start = t * chunk_size;
                thread_args[t].end = (t == N_THREADS - 1) ? CODEBOOK_DIM : (t + 1) * chunk_size;
                thread_args[t].channel = i;
                thread_args[t].quantizer = q;
                thread_args[t].residual = residual;
                thread_args[t].codebook = codebooks[q];
                thread_args[t].results = results;

                pthread_create(&threads[t], NULL, thread_func, &thread_args[t]);
            }

            for (int t = 0; t < N_THREADS; t++) {
                pthread_join(threads[t], NULL);
            }


            int best_codeword = find_best_codeword(results, N_THREADS);
            codeword_ids[i * NUM_QUANTIZERS + q] = best_codeword;


            for (int j = 0; j < N_COL; j++) {
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

    MatrixPrint(codeword_ids, "", N_CHANNELS, NUM_QUANTIZERS);

    free(residual);
    free(codeword_ids);
    free(results);

    printf("Done!\n");
    return 0;
}
