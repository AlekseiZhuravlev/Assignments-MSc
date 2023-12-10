// Exercixe 6.2: daxpy
// E.Suarez, FZJ/UniBonn (2023.11.10)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Measure the runtime
struct timespec start, end;

double time_daxpy(double a, double *x, double *y, long int N) 
{
    int niter = 0;  //Number of iterations needed to gurantee good time measuremements
    double runtime = 0.0;  //Measured runtime
    double *z = (double*) malloc(N*sizeof(double)); 
    double runtime_f = 0.0;  //Measured runtime
    clock_t start_time, curr_time;  //Start and current time
      

    /* --- TODO
     * Implement the main loop of the daxpy function: z = a*x + y
     * For good timing precision, you should:
     *  - Do a "warm up" run that is not taken into account for the timing.
     *  - Run daxpy a number of times (niter) so that total measured runtime is at least 1 microsecond.
     *  - Calculate then the runtime of a single iteration dividing the measured time by niter.
     */

    for (int i=0; i<N; i++)
        z[i] = a*x[i] + y[i];

    curr_time = clock();
    while (curr_time - start_time < 1e-6)
    {
        for (int i=0; i<N; i++)
            z[i] = a*x[i] + y[i];
        curr_time = clock();
    }
    niter = 0;
    start_time = clock();
    while (curr_time - start_time < 1e-6)
    {
        for (int i=0; i<N; i++)
            z[i] = a*x[i] + y[i];
        curr_time = clock();
        niter++;
    }
    runtime_f = (double)(curr_time - start_time) / CLOCKS_PER_SEC;
    runtime_f = runtime_f / niter;
    printf("niter = %d\n", niter);

    

    // dsd
    /* Start time stamp */
    clock_gettime(CLOCK_MONOTONIC, &start); 
    
    #pragma omp parallel for
    for (int n=0; n<niter; n++)
    {
        for (int i=0; i<N; i++)
            z[i] = a*x[i] + y[i];
    }

    /* End timestap */
    clock_gettime(CLOCK_MONOTONIC, &end);
    runtime = (end.tv_nsec - start.tv_nsec) / 1.e3 + (end.tv_sec  - start.tv_sec)*1.e6;  //in microsec
    
    /* --- TODO: Return the result runtime in seconds */
    runtime = runtime / 1.e6;
    runtime = runtime / niter;
    
    free(z);
    return runtime;
}


int main(int argc, char* argv[])
{
    /* Call this program giving the length of the vectors via command line */
   
    /* Define variables */
    long int size = 1024;  //size of vector given via command line, default 1024
    double a = 0.0;        //constant
    double *x;             //vector x
    double *y;             //vector y 
    double rtime = 0.0;    //runtime [sec]
    double t_best = 1000.0;   //best timing [sec]
    double t_worst = 0.0;   //worst timing [sec]
    
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


  
    /* --- TODO Allocate x, y vectors */
    x = (double*) malloc(size*sizeof(double));
    y = (double*) malloc(size*sizeof(double));
    
    
    /* Initialize a, x, and y with random numbers between 0 and 1 */
    srand(12);
    a = (double)rand()/(double)RAND_MAX;
    for (long int i=0; i<size; i++)
    {
        x[i] = (double)rand()/(double)RAND_MAX;
        y[i] = (double)rand()/(double)RAND_MAX;
    }
    
    /* --- TODO 
     * Run the daxpy timing function  
     * Run it 10 times and collect the best and worst timing values
     */
    //pragma omp parallel for before your for loops

    

    for (int i=0; i<10; i++)
    {
        rtime = time_daxpy(a, x, y, size);
        if (rtime < t_best)
            t_best = rtime;
        if (rtime > t_worst)
            t_worst = rtime;
    }

  
    /* --- TODO: Calculate the runtime as the best time */
    rtime = t_best;

  
    /* --- TODO
     * Calculate now the values:
     * flops: number of floating point operations in daxpy (for one single run)
     * bytes: number of bytes transfered to calculate daxpy (for one single run)
     * perf: the performance mesured in FLOPS/sec 
     * bw: the memory bandwidth caculated in Bytes/sec
     */
    flops = 2*size;
    bytes = 3*size*sizeof(double);
    perf = flops / rtime;
    bw = bytes / rtime;


  

    /* Print the output
     *  Size of the vector (number of elements)
     *  Memory bandwidth used [GB/s]
     *  Runtime [msec]
     */
    printf("%10ld  %10.3lf  %10.3lf\n", size, bw/1e9, rtime*1e3);
    
    free (x);
    free (y);
    return 0;
}
