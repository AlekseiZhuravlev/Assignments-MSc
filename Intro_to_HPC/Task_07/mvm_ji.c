/* Dense Matrix-matrix Multiplication
 * E.Suarez (FZJ/UBonn, 2023)
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* For LONG_MAX */
#include <limits.h>
#include <omp.h>

/* Measure the runtime */
struct timespec start, end;


/* Naive implementation, iterate over columns */
double time_mvm(double ** restrict A, double * restrict x, double * restrict y, long int N) 
{
    double runtime = 0.0;  //Measured runtime


    /* --- TODO: Measure time only in the internal loop */
    
     /* --- TODO
     * Implement the main loop of the mvm function: y = y + A *x
     *  - Iterating over columns (inner loop over rows, outer loop over columns)
     */

    double start 

    // start = clock();

    // add pragma omp parallel for

    #pragma omp parallel for

    for (long int i=0; i<N; i++)
    {
        // record the time at the beginning of the loop
        
        // clock_gettime(CLOCK_MONOTONIC, &start);

        for (long int j=0; j<N; j++)
        {
            y[i] += A[i][j] * x[j];
        }
    }

    runtime = (double)(clock() - start) / CLOCKS_PER_SEC;    
    
    /* --- TODO  
     * Return the final runtime in seconds
     */
    
    return runtime;
}


int main(int argc, char* argv[])
{
    /* Call this program giving the length of the vectors via command line */
   
    /* Define variables */
    long int size = 1024;  //size of vector given via command line, default 1024
    double a = 0.0;        //constant
    double *x;             //vector x
    double *y;             //vector y (result)
    double **A;            //matrix A
    double rtime = 0.0;    //runtime [sec]
    
    long int flops = 0.0;  //floating point operations executed
    long int bytes = 0.0;  //bytes  transfered
    double bw = 0.0;     //memory bandwidth    
    double perf = 0.0;   //performance (flops/s)

    
    /* Collect the size of the vector from command line */
    if(argc > 1)
        //convert input text parameter to integer
        size = (long int) atol(argv[1]);
    else
        printf("Missing argument. Default vector size = %ld\n", size);

  
    /* --- TODO: 
     *  Allocate vectors and matrix 
     */
    x = (double*) malloc(size * sizeof(double));
    y = (double*) malloc(size * sizeof(double));
    A = (double**) malloc(size * sizeof(double*));
    for (long int i=0; i<size; i++)
    {
        A[i] = (double*) malloc(size * sizeof(double));
    }
 
    /* Initialize A, x, and y */
    for (long int i=0; i<size; i++)
    {
        x[i] = (double) i / LONG_MAX;
        y[i] = 0.0;
        for (long int j=0; j<size; j++)
        {
            A[i][j] = (double) i / LONG_MAX;
            //printf("A[%3ld][%3ld]=%10.5lf | x[%3ld]=%10.5lf | y[%3ld]=%10.5lf \n", i, j, A[i][j], i, x[i], i, y[i]);
        }
    }
    
    /* --- TODO 
     * In 7.3.a to 7.3.e: Run the mvm timing function  only once 
     * In 7.3.f to 7.3: Run the mvm timing function 5 times and save the best timing into "rtime"
     *   - Hint: remember to re-initialize the vector "y" between iterations
     */

    // run 5 times, save the best timing into "rtime"
    double t_best = time_mvm(A, x, y, size);

    for (int i=0; i<4; i++)
    {
        rtime = time_mvm(A, x, y, size);
        if (rtime < t_best)
            t_best = rtime;
    }
    rtime = t_best;

  
    /* --- TODO
     * Clculate now the values:
     * flops: number of floating point operations in mvm (for one single run)
     * bytes: number of bytes transfered to calculate mvm (for one single run)
     * perf: the performance mesured in FLOPS/sec 
     * bw: the memory bandwidth caculated in Bytes/sec
     */

    flops = 2*size*size;
    bytes = 3*size*size*sizeof(double);
    perf = flops / rtime;
    bw = bytes / rtime;
    
    /* Print the output
     *  Size of the vector (number of elements)
     *  Flops calculated
     *  Bytes transfered
     *  Memory bandwidth used [GB/s]
     *  Runtime [msec]
     */
    
    /* Size FLOP Bytes BW[GB/s] Runtime[ms] */
    printf("%10ld  %10ld  %10ld  %10.3lf  %10.3lf\n", size, flops, bytes, bw/1e9, rtime*1e3);
    
    free (x);
    free (y);
    free (A);
    return 0;
}

