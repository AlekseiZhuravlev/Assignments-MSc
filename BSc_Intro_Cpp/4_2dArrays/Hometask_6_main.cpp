#include "pch.h"

#define SIZE_X 6

#define SIZE_Y 5

void main_hometask()

{
	double* Create_matrix(int size1);
	void Print_matrix(double* matrix, int size);
	double* Multiply_elements(double* matrix1, double* matrix2, int size);
	double** Create_matrix_2d(int sizex, int sizey);
	void Print_matrix_2d(double** matrix, int sizex, int sizey);
	double** Multiply_elements_2d(double** matrix1, double** matrix2, int sizex, int sizey);


	//For current seminar

	double Sqr_matrix(double matrix1);
	double Sqrt_matrix(double matrix1);
	double Asin_matrix(double matrix1);
	double * F_a(double* matrix, double(*matrix1[6])(double));

	double(*matrix1[6])(double);

	for (int i = 0, k = 0; i < 6; i++) {
		k = (int)rand() % 3;
		switch (k) {
		case 0: matrix1[i] = Sqr_matrix;
			cout << "Sqr" << endl;
			break;
		case 1: matrix1[i] = Sqrt_matrix;
			cout << "Sqrt" << endl;
			break;
		case 2: matrix1[i] = Asin_matrix;
			cout << "Asin" << endl;
			break;

		}
		
	}
	cout << endl;

	double* a = Create_matrix(6);
	Print_matrix(a, 6);

	a = F_a(a, matrix1);
	Print_matrix(a, 6);
}