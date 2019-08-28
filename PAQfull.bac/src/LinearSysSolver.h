///*
// * LinearSysSolver.h
// *
// *  Created on: Jul 8, 2013
// *      Author: adm85
// */
//
//#ifndef LINEARSYSSOLVER_H_
//#define LINEARSYSSOLVER_H_
//
//#include <vector>
//#include "cublas_v2.h"
//using namespace std;
//
//class LinearSysSolver {
//public:
//	LinearSysSolver();
//	virtual ~LinearSysSolver();
//	void solveSystem(cuComplex* A, int M_A, int N_A, cuComplex* B, int M_B, int N_B);
//private:
//	void getLUDecomposition(cuComplex* x, int M, int N, cuComplex* LUMat, int* ipiv, int ipivLength);
//	void swapPivotRows(cuComplex* x, int M, int N, int* ipiv, int ipivLength);
//	void cublasSolveLinearSystem(cuComplex* A, int M, int N, cuComplex* B, int M_B, int N_B);
//	cuComplex* multiplyMatrices(cuComplex* A, int M_A, int N_A, cuComplex* B, int M_B, int N_B);
//	void columnMajorPrintArray(cuComplex* x, int M, int N);
//};
//
//#endif /* LINEARSYSSOLVER_H_ */
