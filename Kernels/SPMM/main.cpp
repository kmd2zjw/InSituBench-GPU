/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization

//------------------
#include <VaryTypeAndOperator.h>
#include <PrintOutput.h>

MultiLineExternalVariablesMacros
//#define IndexType long long int
#define IndexType int
const char *sSDKname     = "conjugateGradient";

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(IndexType *I, IndexType *J, float *val, IndexType N,IndexType nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand()/RAND_MAX + 10.0f;
    val[1] = (float)rand()/RAND_MAX;
    int start;

    for (IndexType i = 1; i < N; i++)
    {
        if (i > 1)
        {
            I[i] = I[i-1]+3;
        }
        else
        {
            I[1] = 2;
        }

        start = (i-1)*3 + 2;
        J[start] = i - 1;
        J[start+1] = i;

        if (i < N-1)
        {
            J[start+2] = i + 1;
        }

        val[start] = val[start-1];
        val[start+1] = (float)rand()/RAND_MAX + 10.0f;

        if (i < N-1)
        {
            val[start+2] = (float)rand()/RAND_MAX;
        }
    }

    I[N] = nz;
}
void genNZRandomly(IndexType *I, IndexType *J, float *val, IndexType numRow, IndexType numCol, IndexType  nz,  float percentage, unsigned seed)
{
    I[0] = 0, J[0] = 0;
    int start;
    IndexType actualNZ=0;
    srand(seed);
    IndexType NZ_perRow=1+numCol*percentage;
    printf("number of non zero valuues is %llu\n", (long int) nz);
    for (IndexType i = 0; i < numRow; i++)
    {
    	I[i]=actualNZ;
    	IndexType tIndex=0;
        for (IndexType j = 0; j < NZ_perRow; j++)
        {
        		int temp= rand () %(numCol/NZ_perRow)+1;
        		tIndex=tIndex+temp;
        		if(actualNZ<nz) //(J==N && nz_inEachCo==0)
        		{
        			val[actualNZ]= (float)rand()/RAND_MAX;
        			J[actualNZ]=j;
        			actualNZ++;
        		}
        }

    }
    I[numRow]=actualNZ;
    if(nz!=actualNZ){
    		printf("error:\n");
    		exit(1);
    }
}

int main(int argc, char **argv)
{
	//C = alpha ∗ A ∗ B + beta ∗ D

	IndexType numRowA = 0, numColB = 0, numRowB=0, numColA=0, nnzA = 0,nnzB = 0, nnzC = 0,nnzD = 0  ;
	IndexType *IA = NULL, *JA = NULL, *IB = NULL, *JB = NULL,
			*IC = NULL, *JC = NULL, *ID = NULL, *JD = NULL;
    float *valA = NULL, *valB = NULL, *valC = NULL,  *valD = NULL;
    const float tol = 1e-5f;
    const int max_iter = 10000;
    //float *x;
    //float *rhs;
    //float a, b, na, r0, r1;
    int *d_colA, *d_rowA, *d_colB, *d_rowB, *d_colC, *d_rowC, *d_colD, *d_rowD;
    float *d_valA, *d_valB,*d_valC,*d_valD ;
    //float  *d_x, dot;
    //float *d_r, *d_p, *d_Ax;
    //int k;
    float alpham1;
    float alpha, beta;

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);

    if (devID < 0)
    {
        printf("exiting...\n");
        exit(EXIT_SUCCESS);
    }
    float NZ_percentage=0.01;
    if (checkCmdLineFlag(argc, (const char **)argv, "percentage"))
    {
    		NZ_percentage = getCmdLineArgumentFloat(argc, (const char **)argv, "percentage");
    }
    printf("NZ_percentage is %f \n",NZ_percentage);
    bool diag=checkCmdLineFlag(argc, (const char **)argv, "diag");

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    if (version < 0x11)
    {
        printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
    numRowA = numColB = numColA=10000000;//1048576;
	//MxK  KxN
    if (checkCmdLineFlag(argc, (const char **)argv, "NumRowA"))
     {
     		numRowA= numColB = numColA= getCmdLineArgumentInt(argc, (const char **)argv, "NumRowA");
     }
    if (checkCmdLineFlag(argc, (const char **)argv, "NumColA"))
     {

		numColA=getCmdLineArgumentInt(argc, (const char **)argv, "NumColA");
     }
    if (checkCmdLineFlag(argc, (const char **)argv, "NumColB"))
     {

		numColB=getCmdLineArgumentInt(argc, (const char **)argv, "NumColB");
     }

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
    numRowB=numColA;
    noWarmUp=checkCmdLineFlag(argc, (const char **)argv, "NoWarmUp");
    /* Generate a random tridiagonal symmetric matrix in CSR format */
    //C = alpha ∗ A ∗ B + beta ∗ D
    if(diag){
    		nnzA= nnzB= nnzD=(numColB-2)*3 + 4;
    }else{
    		nnzA=(float)(numRowA)*(float)(numColA)*NZ_percentage ;
    		nnzB=(float)(numRowB)*(float)(numColB)*NZ_percentage ; //
    		nnzD=(float)(numRowA)*(float)(numColB)*NZ_percentage ;
    		printf("nnzA is %d nnzB is %d nnzD is %d percentage is %f\n", nnzA,nnzB,nnzD,NZ_percentage);
    }
    //memory allocations
    IA = (IndexType *)malloc(sizeof(IndexType)*(numRowA+1));
    JA = (IndexType *)malloc(sizeof(IndexType)*nnzA);
    valA = (float *)malloc(sizeof(float)*nnzA);
    //-----------
    IB = (IndexType *)malloc(sizeof(IndexType)*(numRowB+1));
    JB = (IndexType *)malloc(sizeof(IndexType)*nnzB);
    valB = (float *)malloc(sizeof(float)*nnzB);

    //-------------
    ID = (IndexType *)malloc(sizeof(IndexType)*(numRowA+1));
    JD = (IndexType *)malloc(sizeof(IndexType)*nnzD);
    valD = (float *)malloc(sizeof(float)*nnzD);
     //--------------
    if(diag){
    		genTridiag(IA, JA, valA, numColB, nnzA);
    		genTridiag(IB, JB, valB, numColB, nnzB);
    		genTridiag(ID, JD, valD, numColB, nnzD);
    }else{
    		genNZRandomly(IA, JA, valA, numRowA,numColA, nnzA, NZ_percentage,1000);
    		genNZRandomly(IB, JB, valB, numColA,numColB, nnzB, NZ_percentage,5);
    		genNZRandomly(ID, JD, valD, numRowA,numColB, nnzD, NZ_percentage,200);
    }


    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    checkCudaErrors(cublasStatus);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    checkCudaErrors(cusparseStatus);

    cusparseMatDescr_t descrA = 0, descrB = 0, descrC = 0, descrD = 0;
    cusparseStatus = cusparseCreateMatDescr(&descrA);
    checkCudaErrors(cusparseStatus);
    cusparseStatus = cusparseCreateMatDescr(&descrB);
    checkCudaErrors(cusparseStatus);
    cusparseStatus = cusparseCreateMatDescr(&descrC);
    checkCudaErrors(cusparseStatus);
    cusparseStatus = cusparseCreateMatDescr(&descrD);
    checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);

    cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO);

    cusparseSetMatType(descrC,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);

    cusparseSetMatType(descrD,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrD,CUSPARSE_INDEX_BASE_ZERO);


    checkCudaErrors(cudaMalloc((void **)&d_colA, nnzA*sizeof(IndexType)));
    checkCudaErrors(cudaMalloc((void **)&d_rowA, (numRowA+1)*sizeof(IndexType)));
    checkCudaErrors(cudaMalloc((void **)&d_valA, nnzA*sizeof(float)));
    //-----------
    checkCudaErrors(cudaMalloc((void **)&d_colB, nnzB*sizeof(IndexType)));
    checkCudaErrors(cudaMalloc((void **)&d_rowB, (numColA+1)*sizeof(IndexType)));
    checkCudaErrors(cudaMalloc((void **)&d_valB, nnzB*sizeof(float)));

      //---------
    checkCudaErrors(cudaMalloc((void **)&d_colD, nnzD*sizeof(IndexType)));
    checkCudaErrors(cudaMalloc((void **)&d_rowD, (numRowA+1)*sizeof(IndexType)));
    checkCudaErrors(cudaMalloc((void **)&d_valD, nnzD*sizeof(float)));
       //--------------
    cudaMemcpy(d_colA, JA, nnzA*sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowA, IA, (numRowA+1)*sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valA, valA, nnzA*sizeof(float), cudaMemcpyHostToDevice);

    //-------------------
    cudaMemcpy(d_colB, JB, nnzB*sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowB, IB, (numColA+1)*sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valB, valB, nnzB*sizeof(float), cudaMemcpyHostToDevice);
    //------------------
    cudaMemcpy(d_colD, JD, nnzD*sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowD, ID, (numRowA+1)*sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valD, valD, nnzD*sizeof(float), cudaMemcpyHostToDevice);
    //-----------------


    //cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);

    alpha = 1.0;
    alpham1 = -1.0;
    //beta = 0.0;
    beta = 0.4;
    //r0 = 0.;


#ifndef METRIC_NOT_RUN_OTHER_EVENTS
    cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// Record the start event
	checkCudaErrors(cudaEventRecord(start, NULL));

#endif
	for (int j = 0; j < nIter; j++)
	{
		int baseC;
		csrgemm2Info_t info = NULL;
		size_t bufferSize;
		void *buffer = NULL;
		// nnzTotalDevHostPtr points to host memory
		int *nnzTotalDevHostPtr = &nnzC;
		//double alpha = -1.0;
		//double beta  =  1.0;
		cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);

		// step 1: create an opaque structure
		cusparseCreateCsrgemm2Info(&info);

		// step 2: allocate buffer for csrgemm2Nnz and csrgemm2
		cusparseScsrgemm2_bufferSizeExt(cusparseHandle, numRowA, numColB, numColA, &alpha,
		    descrA, nnzA, d_rowA, d_colA,
		    descrB, nnzB, d_rowB, d_colB,
		    &beta,
		    descrD, nnzD, d_rowD, d_colD,
		    info,
		    &bufferSize);
		cudaMalloc(&buffer, bufferSize);

		// step 3: compute csrRowPtrC
		cudaMalloc((void**)&d_rowC, sizeof(int)*(numRowA+1));
		cusparseXcsrgemm2Nnz(cusparseHandle, numRowA, numColB, numColA,
		        descrA, nnzA, d_rowA, d_colA,
		        descrB, nnzB, d_rowB, d_colB,
		        descrD, nnzD, d_rowD, d_colD,
		        descrC, d_rowC, nnzTotalDevHostPtr,
		        info, buffer );
		if (NULL != nnzTotalDevHostPtr){
		    nnzC = *nnzTotalDevHostPtr;
		}else{
		    cudaMemcpy(&nnzC, d_rowC+numRowA, sizeof(int), cudaMemcpyDeviceToHost);
		    cudaMemcpy(&baseC, d_rowC, sizeof(int), cudaMemcpyDeviceToHost);
		    nnzC -= baseC;
		}

		// step 4: finish sparsity pattern and value of C
		cudaMalloc((void**)&d_colC, sizeof(int)*nnzC);
		cudaMalloc((void**)&d_valC, sizeof(float)*nnzC);
		// Remark: set csrValC to null if only sparsity pattern is required.
		//C = alpha * A *B + beta * D

		cusparseScsrgemm2(cusparseHandle, numRowA, numColB, numColA, &alpha,
		        descrA, nnzA, d_valA, d_rowA, d_colA,
		        descrB, nnzB, d_valB, d_rowB, d_colB,
		        &beta,
		        descrD, nnzD, d_valD, d_rowD, d_colD,
		        descrC, d_valC, d_rowC, d_colC,
		        info, buffer);

		// step 5: destroy the opaque structure
		cusparseDestroyCsrgemm2Info(info);

	}
	float msecTotal = 0.0f;
#ifndef METRIC_NOT_RUN_OTHER_EVENTS
	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, NULL));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
#endif
    double msec = msecTotal / nIter;
     double gigaFlops=0;
     sizeInGBytes= (2*sizeof(IndexType)*(numRowA)+sizeof(IndexType)*(numColB)+sizeof(IndexType)*(nnzA+nnzB+nnzD)+sizeof(float)*(nnzA+nnzB+nnzD))* 1.0e-9;
     if(msec!=0){
     	gigaProcessedInSec=( sizeInGBytes / (msec / 1000.0f));
     }
     outPutSizeInGBytes=(sizeof(IndexType)*(numRowA)+sizeof(IndexType)*(nnzC)+sizeof(float)*nnzC)*1.0e-9;
     printf("numRowA= %d numColA=%d numColB=%d \n", numRowA, numColA, numColB);
     timeInMsec=msec;
     printOutput();
     printf("nIter %d\n", nIter);

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    free(IA);
    free(JA);
    free(valA);
    //-------------
    free(IB);
    free(JB);
    free(valB);
    //------------
    /*
     free(IC);
     free(JC);
     free(valC);
     */
     //------------
     free(ID);
    free(JD);
    free(valD);
    //-------------
    cudaFree(d_colA);
    cudaFree(d_rowA);
    cudaFree(d_valA);
    //-------------
    cudaFree(d_colB);
    cudaFree(d_rowB);
    cudaFree(d_valB);
    //-------------
    cudaFree(d_colC);
    cudaFree(d_rowC);
    cudaFree(d_valC);
    //-------------
    cudaFree(d_colD);
    cudaFree(d_rowD);
    cudaFree(d_valD);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    //printf("Test Summary:  Error amount = %f\n", err);
    //exit((k <= max_iter) ? 0 : 1);
}
