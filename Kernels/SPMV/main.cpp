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

///--------------Added for reading from file
#include <fstream>
#include <iostream>
//#include <string.h>
#include <sstream>
#include <limits>
using namespace std;


//---------------

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
//#define METRIC_NOT_RUN_OTHER_EVENTS

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
        }//
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
void calculateRequiredMemory(char * sparseFilename,  IndexType& numRow,IndexType& numCol,IndexType& MinNumCol, IndexType& MinNzPerRow, IndexType& MaxNzPerRow, IndexType& nz, float & NZ_percentage){

	string line, entry;	
	ifstream sparseFile(sparseFilename);
	nz=0;
	IndexType row_ind=0;
	numCol=0;
	if(sparseFile.is_open()){
		IndexType col_ind=0;
   		while(getline(sparseFile, line)){   // get a whole line
       			std::stringstream s1(line);
				IndexType NzPerRow=0;
        			while(getline(s1, entry, ' ')){
					
        				std::stringstream s2(entry);
					int colonIndex=entry.find(':');
					if (entry.find(':') ==string::npos){
						col_ind=0;
					}else{
						col_ind =atoi(entry.substr(0, colonIndex).c_str());
						NzPerRow++;
						nz++;       			
					}
				}
        			if(col_ind>numCol){
        				numCol=col_ind;
        			}
				if(MinNumCol==0){
					MinNumCol=col_ind;
				}else{
					if(MinNumCol>col_ind)
						MinNumCol=col_ind;
				}
				if(MinNzPerRow > NzPerRow)
					MinNzPerRow=NzPerRow;
				if(MaxNzPerRow <NzPerRow)
					MaxNzPerRow=NzPerRow;
        		row_ind++;
  		 }
	}
	numRow=row_ind;
	NZ_percentage=float(nz)/(float(numRow)*(numCol));
	if(numRow<1){
		printf("error: the file has no line %s\n", sparseFilename);
		exit(1);
	}
	printf(" numRow is %d, numCol is %d ,nz is %d, NZ_percentage is %f \n", numRow,numCol, nz,NZ_percentage);
}
void genByReadingAfile(char * sparseFilename, IndexType *I, IndexType *J, float *val, IndexType numRow,IndexType  nz)
{
    //IndexType col_ind,row_ind;
	string line, entry;
	float value;
	IndexType index;
	ifstream sparseFile(sparseFilename);
	IndexType actualNz=0;
	IndexType row_ind=0;
	if(sparseFile.is_open()){
   		while(getline(sparseFile, line)){   // get a whole line
       			std::stringstream s1(line);
       			IndexType col_ind=0;
       			I[row_ind]=actualNz;
        			while(getline(s1, entry, ' ')){
        				std::stringstream s2(entry);
					int colonIndex=entry.find(':');
					if (entry.find(':') ==string::npos){
						index=0;
					}else{
						index =atoi(entry.substr(0, colonIndex).c_str());
						value =atof( entry.substr(entry.find(':')+1).c_str());
						
						val[actualNz]=value;
						J[actualNz]=index;
						col_ind++;
						actualNz++;
					}
					
					//cout<<index<<":"<<value<<endl;
						// You now have separate entites here
       			}
        		row_ind++;
  		 }
	}
	I[row_ind]=row_ind;
	
	if(numRow!=row_ind or actualNz!=nz){
		printf("error: number of row or nz does not match %d %d vs %d %d \n", numRow,nz, row_ind,actualNz);
		exit(1);
	}
}

int main(int argc, char **argv)
{
    IndexType numRow = 0, numCol = 0, MinNumCol=std::numeric_limits<IndexType>::max(), nz = 0, *I = NULL, *J = NULL;
    IndexType MaxNzPerRow=std::numeric_limits<IndexType>::min();
    IndexType MinNzPerRow=std::numeric_limits<IndexType>::max();

    float *val = NULL;
    const float tol = 1e-5f;
    const int max_iter = 10000;
    float *x,*ax;
    //float *rhs;

    //y = α ∗ op ( A ) ∗ x + β ∗ y
    //A is an m×n sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); x and y are vectors; α  and  β are scalars; and

    float a, b, na, r0, r1;
    int *d_col, *d_row;
    float *d_val, *d_x, dot;
    float *d_r, *d_p, *d_Ax;
    int k;
    float alpha, beta, alpham1;
    char* sparseFileName;
    bool generateByFile=false;

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
    bool diag=checkCmdLineFlag(argc, (const char **)argv, "diag");
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
    noWarmUp=checkCmdLineFlag(argc, (const char **)argv, "NoWarmUp");
    if (checkCmdLineFlag(argc, (const char **)argv, "sparseFile")){
		
    		generateByFile=getCmdLineArgumentString(argc,(const char **) argv,"sparseFile", &sparseFileName );
		printf(" read data from %s \n",sparseFileName);
    }
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
    numRow = numCol = 10000000;//1048576;

    /* Generate a random tridiagonal symmetric matrix in CSR format */
  //y = α ∗ op ( A ) ∗ x + β ∗ y
//A is an m×n sparse matrix that is defined in CSR storage format by the three arrays csrValA, csrRowPtrA, and csrColIndA); x and y are vectors; α  and  β are scalars; and
	if(generateByFile==true){
		calculateRequiredMemory(sparseFileName,numRow, numCol, MinNumCol,MinNzPerRow,MaxNzPerRow, nz, NZ_percentage);

	}else{
	    if (checkCmdLineFlag(argc, (const char **)argv, "NumRow"))
	     {
	     		numRow= numCol = getCmdLineArgumentInt(argc, (const char **)argv, "NumRow");
	     }
		if(checkCmdLineFlag(argc, (const char **)argv, "NumCol"))
		{
			numCol=getCmdLineArgumentInt(argc, (const char **)argv, "NumCol");
		}
		   if(diag){
		    		nz = (numCol-2)*3 + 4;
		    }else{
		    		nz=(NZ_percentage * numRow*numCol);

		    }
	}
	//printf(" before allocation: numCol is %d, numRow is %d ,nz is %d, NZ_percentage is %f \n", numCol,numRow, nz,NZ_percentage);

    I = (IndexType *)malloc(sizeof(IndexType)*(numRow+1));
    J = (IndexType *)malloc(sizeof(IndexType)*nz);
    val = (float *)malloc(sizeof(float)*nz);
	if(generateByFile==true){
		 genByReadingAfile(sparseFileName, I, J, val,  numRow,nz);
		 //NZ_percentage=nz/(numCol*numRow);
		 /*
		 for(int t1=0; t1<10;t1++)
		 {
			 printf("I[%d]= %d \n", t1, I[t1]);
			 for(int t2=0; t2<10;t2++)
			 {
				 printf("j[%d]= %d \t", t2, J[t2]);
				 printf("val[%d]= %d \t", t2, val[t2]);

			 }

		 }
		 */
	}else{
		   if(diag){
			   genTridiag(I, J, val, numCol, nz);
		    }else{
		    		genNZRandomly(I, J, val, numRow,numCol, nz, NZ_percentage, 5);
		    }
	}

	printf(" after allocation: numCol is %d, numRow is %d ,nz is %d, NZ_percentage is %f \n", numCol,numRow, nz,NZ_percentage);

    x = (float *)malloc(sizeof(float)*numCol);
    ax= (float *)malloc(sizeof(float)*numRow);
    //rhs = (float *)malloc(sizeof(float)*N);

    for (int i = 0; i < numCol; i++)
    {
        //rhs[i] = 1.0;
        //x[i] = 0.0;
        x[i] = (float)rand()/RAND_MAX + 10.0f;;
    }
    for (int i = 0; i < numRow; i++)
     {

    		ax[i] = (float)rand()/RAND_MAX + 10.0f;;
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

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    checkCudaErrors(cusparseStatus);

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(IndexType)));
    checkCudaErrors(cudaMalloc((void **)&d_row, (numRow+1)*sizeof(IndexType)));
    checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_x, numCol*sizeof(float)));
    //checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
    //checkCudaErrors(cudaMalloc((void **)&d_p, numCol*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Ax, numRow*sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_col, J, nz*sizeof(IndexType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row, I, (numRow+1)*sizeof(IndexType), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, x, numCol*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_Ax, ax, numRow*sizeof(float), cudaMemcpyHostToDevice));
    //cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);

    alpha = 1.0;
    alpham1 = -1.0;
    //beta = 0.0;
    beta = 0.4;
    r0 = 0.;


#ifndef METRIC_NOT_RUN_OTHER_EVENTS
    cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	// Record the start event
	checkCudaErrors(cudaEventRecord(start, NULL));
#endif
	////y = α ∗ op ( A ) ∗ x + β ∗ y
	for (int j = 0; j < nIter; j++)
	{
		//printf("iter is %d\n", j);

		cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, numRow, numCol, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

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
     sizeInGBytes= (sizeof(IndexType)*(numRow+1)+sizeof(IndexType)*nz+sizeof(float)*nz+sizeof(float)*numCol)* 1.0e-9;
     if(msec!=0){
     	gigaProcessedInSec=( sizeInGBytes / (msec / 1000.0f));
     }
     outPutSizeInGBytes=sizeof(float)*numCol*1.0e-9;
     timeInMsec=msec;
     printOutput();
     printf("nIter %d\n", nIter);

     if(generateByFile==true){
		 printf(" MaxNzPerRow is %d\n", MaxNzPerRow);
		 printf(" MinNzPerRow is %d\n", MinNzPerRow);
     }
    /*
    cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

    k = 1;

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
        else
        {
            cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
        cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaThreadSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    float rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i+1]; j++)
        {
            rsum += val[j]*x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }
/*/
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    free(I);
    free(J);
    free(val);
    free(x);
//    free(rhs);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    //printf("Test Summary:  Error amount = %f\n", err);
    //exit((k <= max_iter) ? 0 : 1);
}
