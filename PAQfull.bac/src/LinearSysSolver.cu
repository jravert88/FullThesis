///*
// * LinearSysSolver.cpp
// *
// *  Created on: Jul 8, 2013
// *      Author: adm85
// */
//
//#include <vector>
//#include <iostream>
//#include <time.h>
//#include "LinearSysSolver.h"
//#include "cublas_v2.h"
//#include "cula.h"
//
//
//LinearSysSolver::LinearSysSolver()
//{
//	// TODO Auto-generated constructor stub
//
//}
//
//LinearSysSolver::~LinearSysSolver()
//{
//	// TODO Auto-generated destructor stub
//}
//
///**
// * Solves A*x=B for x. The result is stored in the vector pointed to by B.
// */
//void LinearSysSolver::solveSystem(cuComplex* A, int M_A, int N_A, cuComplex* B, int M_B, int N_B) {
//	//Get the LU Factorization
//	cuComplex* LUMat = new cuComplex[M_A*N_A];
//	int ipivLength = N_A;
//	int* ipiv = new int[ipivLength];
//	getLUDecomposition(A, M_A, N_A, LUMat, ipiv, ipivLength);
//
//	//Calculate P*b
//	swapPivotRows(B, M_B, N_B, ipiv, ipivLength);
//
//	//Solve the system. The result will be stored in B
//	cublasSolveLinearSystem(LUMat, M_A, N_A, B, M_B, N_B);
//
//	//  DEBUG CODE -------
//	//cuComplex* test = multiplyMatrices(xTxInv, N, N, xTx, N, N);
//	cuComplex* test = multiplyMatrices(A, M_A, N_A, B, M_B, N_B);
//	cout << endl << "X * XInv" << endl;
//	columnMajorPrintArray(test, M_A, N_B);
//	delete [] test;
//	//  END DEBUG CODE ---
//
//	delete [] LUMat;
//	delete [] ipiv;
//}
//
//
///**
// * Uses the CULA library to get the LU decomposition of the matrix.
// */
//void LinearSysSolver::getLUDecomposition(cuComplex* x, int M, int N, cuComplex* LUMat, int* ipiv, int ipivLength) {
//
//	culaDeviceFloatComplex* devxTx;
//	culaDeviceInt* devIPIV;
//
//	cudaMalloc(&devxTx, M*N*sizeof(culaDeviceFloatComplex));
//	cudaMalloc(&devIPIV, ipivLength*sizeof(culaDeviceInt));
//	cudaMemcpy(devxTx, x, M*N*sizeof(culaDeviceFloatComplex), cudaMemcpyHostToDevice);
//
//	culaStatus culaStat;
//	culaInitialize();
//
//	culaStat = culaDeviceCgetrf(M, N, devxTx, M, devIPIV);
//	if(culaStat != culaNoError) {
//		cout << "Cula Cgetrf failure" << endl;
//	}
//
//	culaShutdown();
//
//	//LUMat = new cuComplex[M*N];
//	cudaMemcpy(LUMat, devxTx, M*N*sizeof(culaDeviceFloatComplex), cudaMemcpyDeviceToHost);
//	cudaMemcpy(ipiv, devIPIV, ipivLength*sizeof(culaDeviceInt), cudaMemcpyDeviceToHost);
//
////	getL(L, LUMat, M, N);
////
//	cout << "LUMat Inside:" << endl;
//	columnMajorPrintArray(LUMat, M, N);
////
////	getU(U, LUMat, M, N);
////	cout << endl << "U" << endl;
////	columnMajorPrintArray(U, M, N);
//
//	cudaFree(devxTx);
//	cudaFree(devIPIV);
//}
//
///**
// * Using the information from the CULA generated IPIF array,
// * this function swaps rows as appropriate.
// */
//void LinearSysSolver::swapPivotRows(cuComplex* x, int M, int N, int* ipiv, int ipivLength) {
//	//Temporary row vector
//	cuComplex rowVec[N];
//
//	//We use index 1 based ordering because this is what CULA returns
//	for(int i=1; i <= ipivLength; i++) {
//		//Check to see if the row swaps. This happens when element x of the ipif
//		//array is not equal to x. When element x is different, it means that row x
//		//and the row specified in element x swap places.
//		if(ipiv[i-1] != i) {
//			int startIndex = i-1;
//			//Copy the current row into the temporary row vector
//			for(int j = 0; j < N; j++) {
//				rowVec[j].x = x[startIndex+j*M].x;
//				rowVec[j].y = x[startIndex+j*M].y;
//			}
//
//			//Copy the specified row into the current row
//			int specRowStart = ipiv[i-1]-1;
//			for(int j=0; j < N; j++) {
//				x[startIndex+j*M].x = x[specRowStart+j*M].x;
//				x[startIndex+j*M].y = x[specRowStart+j*M].y;
//			}
//
//			//Copy the temp row into the specified row
//			for(int j=0; j < N; j++) {
//				x[specRowStart+j*M].x = rowVec[j].x;
//				x[specRowStart+j*M].y = rowVec[j].y;
//			}
//		}
//	}
//
//}
//
//void LinearSysSolver::cublasSolveLinearSystem(cuComplex* A, int M, int N, cuComplex* B, int M_B, int N_B) {
//	cuComplex* xInv = new cuComplex[M*N_B];
//
//	//Now put L, U, and the I matrix on the GPU
//	cublasStatus_t stat;
//	cublasHandle_t handle;
//
//	cuComplex* devA;
//	cuComplex* devB;
//	cudaMalloc(&devA, M*N*sizeof(cuComplex));
//	cudaMalloc(&devB, M_B*N_B*sizeof(cuComplex));
//
//	stat = cublasCreate(&handle);
//	if(stat != CUBLAS_STATUS_SUCCESS) {
//		cout << "Error in solver" << endl;
//	}
//	stat = cublasSetMatrix(M, N, sizeof(cuComplex), A, M, devA, M);
//	if(stat != CUBLAS_STATUS_SUCCESS) {
//		cout << "Error in solver" << endl;
//	}
//	stat = cublasSetMatrix(M_B, N_B, sizeof(cuComplex), B, M_B, devB, M_B);
//	if(stat != CUBLAS_STATUS_SUCCESS) {
//		cout << "Error in solver" << endl;
//	}
//
//	//Set up Alpha
//	cuComplex alpha;
//	alpha.x = 1;
//	alpha.y = 0;
//
//	//First solve L*y = P*b
//	stat = cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, M, N, &alpha, devA, M, devB, M_B);
//	if(stat != CUBLAS_STATUS_SUCCESS) {
//		cout << "Error solving for y" << endl;
//	}
//
//	//Then solve U*x = y
//	stat = cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N, &alpha, devA, M, devB, M_B);
//	if(stat != CUBLAS_STATUS_SUCCESS) {
//		cout << "Error solving for x" << endl;
//	}
//
//	//Get results, and store them in matrix B
//	cudaMemcpy(B, devB, M*N_B*sizeof(cuComplex), cudaMemcpyDeviceToHost);
//
//	//Free resources
//	cublasDestroy(handle);
//	cudaFree(devA);
//	cudaFree(devB);
//}
//
///**
// * Multiplies two matrices together. Result is stored in B on exit.
// */
//cuComplex* LinearSysSolver::multiplyMatrices(cuComplex* A, int M_A, int N_A, cuComplex* B, int M_B, int N_B) {
//	cudaError_t cudaStat;
//	cublasStatus_t stat;
//	cublasHandle_t handle;
//
//	cuComplex* devA;
//	cuComplex* devB;
//	cuComplex* devC;
//	cuComplex* alpha = new cuComplex;
//	cuComplex* beta = new cuComplex;
//	cuComplex* hostC = new cuComplex[M_A*N_B];
//	alpha->x = 1;
//	alpha->y = 0;
//	beta->x = 0;
//	beta->y = 0;
//
//	cudaStat = cudaMalloc(&devA, M_A*N_A*sizeof(cuComplex));
//	cudaStat = cudaMalloc(&devB, M_B*N_B*sizeof(cuComplex));
//	cudaStat = cudaMalloc(&devC, M_A*N_B*sizeof(cuComplex));
//	if(cudaStat != cudaSuccess) {
//		cout << "Horrible failure!" << endl;
//	}
//
//	stat = cublasCreate(&handle);
//
//	stat = cublasSetMatrix(M_A, N_A, sizeof(cuComplex), A, M_A, devA, M_A);
//	if (stat != CUBLAS_STATUS_SUCCESS) {
//		cout << "Data download A failed" << endl;
//	}
//	stat = cublasSetMatrix(M_B, N_B, sizeof(cuComplex), B, M_B, devB, M_B);
//	if (stat != CUBLAS_STATUS_SUCCESS) {
//		cout << "Data download B failed" << endl;
//	}
//
//	//Perform the multiply.
//	stat = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M_A, N_B, N_A, alpha, devA, M_A, devB, M_B, beta, devC, M_A);
//
//	stat = cublasGetMatrix(M_A, N_B, sizeof(cuComplex), devC, M_A, hostC, M_A);
//	if (stat != CUBLAS_STATUS_SUCCESS) {
//		cout << "Failed to get devC to hostC" << endl;
//		cout << stat << endl;
//	}
//
//	cudaFree(devA);
//	cudaFree(devB);
//	cudaFree(devC);
//	cublasDestroy(handle);
//
//	delete alpha;
//	delete beta;
//	return hostC;
//
//}
//
///**
// * Prints out an array that is stored in column-major order in memory.
// */
//void LinearSysSolver::columnMajorPrintArray(cuComplex* x, int M, int N) {
//	int realIndex;
//	cout << "------------------------------------------------------" << endl;
//	cout << "              Printing Column Order Matrix            " << endl;
//	cout << "------------------------------------------------------" << endl;
//	for(int i=0; i < M; i++) {
//		cout << "Row: " << (i+1) << "    ";
//		for(int j=0; j < N; j++) {
//			realIndex = (M*j)+i;
//			cout << x[realIndex].x;
//			if(x[realIndex].y >= 0) {
//				cout << "+";
//			}
//			cout << x[realIndex].y << "i ";
//		}
//		cout << endl;
//	}
//}
