#include "pch.h"
#include <iomanip>  // IMPORTANT!!!!!!!!!!!!!!!!


double* Create_matrix(int size1) {

	double *matrix1 = new double[size1];
		for (int i = 0; i < size1; i++) {
			matrix1[i] = ((double)rand() / (double)RAND_MAX);
	}

		return matrix1;
}

void Print_matrix(double* matrix, int size) {
	for (int i = 0; i < size; i++) {
		cout << matrix[i] << endl;
	}
	cout << endl;
}

double* Multiply_elements(double* matrix1, double* matrix2, int size) {

	for (int i = 0; i < size; i++) {
		matrix1[i] *= matrix2[i];
	}

	return matrix1;
}

double** Create_matrix_2d(int sizex, int sizey) {

	double **matrix1 = new double*[sizex];

	for (int i = 0; i < sizex; i++) {
		matrix1[i] = new double[sizey];
	}

	for (int i = 0; i < sizex; i++) {
		for (int j = 0; j < sizex; j++) {
			matrix1[i][j] = (1.5 + 1) * ((double)rand() / (double)RAND_MAX) - 1;
		}
	}

	return matrix1;
}

void Print_matrix_2d(double** matrix, int sizex, int sizey) {
	for (int i = 0; i < sizex; i++) {
		for (int j = 0; j < sizey; j++) {
			cout << setw(10) << matrix[i][j] << " ";
			//#include <iomanip>  // IMPORTANT!!!!!!!!!!!!!!!!
		}
		cout << endl;
	}
	cout << endl;
}

double** Multiply_elements_2d(double** matrix1, double** matrix2, int sizex, int sizey) {

	for (int i = 0; i < sizex; i++) {
		for (int j = 0; j < sizey; j++) {
			matrix1[i][j] *= matrix2[i][j];
		}
	}

	return matrix1;
}