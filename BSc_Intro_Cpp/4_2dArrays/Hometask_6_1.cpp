#include "pch.h"
#include <iomanip>  // IMPORTANT!!!!!!!!!!!!!!!!


double* Sqr_matrix(double* matrix1, int size) {

	for (int i = 0; i < size; i++) {
		matrix1[i] *= matrix1[i];
	}

	return matrix1;
}

double* Sqrt_matrix(double* matrix1, int size) {

	for (int i = 0; i < size; i++) {
		matrix1[i] = sqrt(matrix1[i]);
	}

	return matrix1;
}

double* aSin_matrix(double* matrix1, int size) {

	double a = 4; // Number here

	for (int i = 0; i < size; i++) {
		matrix1[i] = sin(matrix1[i]) * a;
	}

	return matrix1;
}