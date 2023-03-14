#include "pch.h"
#include <iomanip>  // IMPORTANT!!!!!!!!!!!!!!!!


double Sqr_matrix(double matrix1) {

		matrix1 *= matrix1;

	return matrix1;
}

double Sqrt_matrix(double matrix1) {

	matrix1 = sqrt(matrix1);

	return matrix1;
}

double Asin_matrix(double matrix1) {

	int a = 4;

	matrix1 = sin (matrix1) * a;

	return matrix1;
}

double * F_a(double* matrix, double(*matrix1[6])(double)) {

	for (int i = 0; i < 6; i++) {
		matrix[i] = matrix1[i](matrix[i]);
	}

	return matrix;
}