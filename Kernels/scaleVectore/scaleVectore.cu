#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
//Modified version of the follwing code foud in stackoveflow
//https://stackoverflow.com/questions/20071454/thrust-gathering-filtering
//There is also this code that you can use
//Added by me
#include <helper_cuda.h>

#include <algorithm>
#include <time.h>
#include <limits.h>

//--------------------------
#include <VaryTypeAndOperator.h>
#include <PrintOutput.h>
#include <thrust/generate.h>
#include <iostream>
/*
template<typename T>
//struct quantize : thrust::unary_function<T, char>


struct quantize 
{
    __host__ __device__
    char operator()(const T &x)
    {
        return (char)(x);
    }
};
__host__ static __inline__ float rand_01()
{
    return ((float)rand()/RAND_MAX);
};
*/

int run(int argc, char **argv)
{
    std::cout << "Loading test!" << std::endl;

    cudaError_t err = cudaSuccess;
    int numElements = 50000;
    if (checkCmdLineFlag(argc, (const char **)argv, "Num"))
    {
        numElements = getCmdLineArgumentInt(argc, (const char **)argv, "Num");

        if (numElements < 0)
        {
            printf("Error: elements must be > 0, elements=%d is invalid\n", numElements);
            exit(EXIT_SUCCESS);
        }
    }
    int N=numElements;
    if (checkCmdLineFlag(argc, (const char **)argv, "nIter"))
    {
    		nIter = getCmdLineArgumentInt(argc, (const char **)argv, "nIter");
    }else{
		#ifndef METRIC_RUN_ONLY_ONCE
			nIter = 30;
		#else
			nIter = 1;
		#endif
	}
    // Print the vector length to be used, and compute its size

     //size_t size = numElements * sizeof(int);
     printf("[Vector filtering of %d elements]\n", numElements);
     noWarmUp=checkCmdLineFlag(argc, (const char **)argv, "NoWarmUp");
    thrust::device_vector<float> col1(N);
    thrust::host_vector<float> h_col1(N);
    for(int i=0 ; i<N; i++){
		h_col1[i]=rand();
	}
//    thrust::generate(col1.begin(), col1.end(), rand_01);
    col1=h_col1;

 
    thrust::device_vector<int> result(N);
     
#ifndef METRIC_NOT_RUN_OTHER_EVENTS
    // run multiple iterations to compute an average sort time

    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));
    checkCudaErrors(cudaEventRecord(start_event, 0));
#endif
    float msecTotal;
    thrust::device_vector<int>::iterator end;

//    float mean=2.5;
    float scale=0.5;

    using namespace thrust::placeholders;
    for ( int t=0;t<nIter;t++){

        //thrust::transform(col1.begin(), col1.end(), result.begin(), quantize<float>());
        thrust::transform(col1.begin(), col1.end(), result.begin(), _1*scale);
    }

    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start_event, stop_event));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));
    int len = end - result.begin();
    double msec = msecTotal / nIter;
     sizeInGBytes= (sizeof(float)*numElements)* 1.0e-9;
     if(msec!=0){
     	gigaProcessedInSec=( sizeInGBytes / (msec / 1000.0f));
     }
     outPutSizeInGBytes=sizeof(int)*numElements/2*1.0e-9;
     timeInMsec=msec;
     printOutput();
     printf("nIter %d\n", nIter);

    thrust::host_vector<int> h_result(10);
    thrust::copy_n(result.begin(), 10, h_result.begin());
    thrust::copy_n(h_result.begin(), 10, std::ostream_iterator<int>(std::cout, "\n"));

    return 0;
}
int main(int argc, char **argv){
	run ( argc, argv);
	checkCudaErrors(cudaDeviceReset());
 	return 0;
}
