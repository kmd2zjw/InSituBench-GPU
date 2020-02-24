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

template<typename T>
struct is_even : thrust::unary_function<T, bool>
{
    __host__ __device__
    bool operator()(const T &x)
    {
        return (x%2)==0;
    }
};


struct is_true : thrust::unary_function<bool, bool>
{
    __host__ __device__
    bool operator()(const bool &x)
    {
        return x;
    }
};

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

    thrust::device_vector<int> col1(N);
    thrust::host_vector<int> h_col1(N);
    thrust::sequence(h_col1.begin(), h_col1.end());
    col1=h_col1;

    //thrust::device_vector<bool> filter(N);
    thrust::device_vector<int> result(N/2+1);
     /* this part is from the stackOverflowCode
    //thrust::transform(col1.begin(), col1.end(), filter.begin(), is_even<int>());
    // filter col1 based on filter
    */
#ifndef METRIC_NOT_RUN_OTHER_EVENTS
    // run multiple iterations to compute an average sort time
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));
    checkCudaErrors(cudaEventRecord(start_event, 0));
#endif
    float msecTotal;
    thrust::device_vector<int>::iterator end;
    for ( int t=0;t<nIter;t++){
    		end = thrust::copy_if(col1.begin(), col1.end(), col1.begin(), result.begin(), is_even<int>());
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
     sizeInGBytes= (sizeof(int)*numElements)* 1.0e-9;
     if(msec!=0){
     	gigaProcessedInSec=( sizeInGBytes / (msec / 1000.0f));
     }
     outPutSizeInGBytes=sizeof(int)*numElements/2*1.0e-9;
     timeInMsec=msec;
     printOutput();
     printf("nIter %d\n", nIter);

    thrust::host_vector<int> h_result(len);
    thrust::copy_n(result.begin(), len, h_result.begin());
    thrust::copy_n(h_result.begin(), 10, std::ostream_iterator<int>(std::cout, "\n"));
    //checkCudaErrors(cudaDeviceReset());

    return 0;
}
int main (int argc, char **argv){
    run(argc, argv);
    checkCudaErrors(cudaDeviceReset());
    return 0;
}
