#include "pch.h"
#include <iomanip>  // IMPORTANT!!!!!!!!!!!!!!!!


double** Create_matrix_triangle(int size) {

	double **matrix1 = new double*[size];

	for (int i = 0; i < size; i++) {
		matrix1[i] = new double[size - i];
	}

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < (size - i); j++) {
			matrix1[i][j] = (1.5 + 1) * ((double)rand() / (double)RAND_MAX) - 1;
		}
	}

	return matrix1;
}

void Print_matrix_triangle(double** matrix, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size - i; j++) {
			cout << setw(10) << matrix[i][j] << " ";
			//#include <iomanip>  // IMPORTANT!!!!!!!!!!!!!!!!
		}
		cout << endl;
	}
	cout << endl;
}

void Delete_matrix_triangle(double** matrix, int size) {
	for (int i = 0; i < size; i++) {
		delete matrix[i];
	}
	delete matrix;
}

double** Add_elements_triangle(double** matrix1, double** matrix2, int size) {

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size - i; j++) {
			matrix1[i][j] += matrix2[i][j];
		}
	}

	return matrix1;
}

double** Multiply_On_Number_triangle(double** matrix, int size) {

	double a = 4;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size - i; j++) {
			matrix[i][j] *= a;
		}
	}

	return matrix;
}

double Find_element_triangle(double** matrix, int size, int i, int j) {

	return matrix[i][j - i];
}

double Find_det_triangle(double** matrix, int size) {

	double a = 1;

	for (int i = 0; i < size; i++) {
		a *= matrix[i][0];
	}
	return a;
}