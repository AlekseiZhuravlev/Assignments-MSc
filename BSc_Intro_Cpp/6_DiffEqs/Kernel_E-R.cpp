#include "pch.h"
#define N_EQS 3
#include <fstream>

void main_kernel() {

	double F_1(double x, double y);
	double F_2(double x, double y);
	double F_3(double x, double y);

	double(*func_array[N_EQS])(double, double);  // Create array of functions

	func_array[0] = F_1;
	func_array[1] = F_2;
	func_array[2] = F_3;

	ofstream file_array[N_EQS];			 // Create array of text files
	file_array[0].open("F1.txt");
	file_array[1].open("F2.txt");
	file_array[2].open("F3.txt");

	double x = 2, h = 0.1, n = 20;		 // Set starting values
	double* y = new double [N_EQS];
	y[0] = 1;
	y[1] = 2;
	y[2] = 3;


	double* Euler(double x, double* y, double(*func_array[N_EQS])(double, double), double h);
	double* R_Cutt(double x, double* y, double(*func_array[N_EQS])(double, double), double h);

	for (int i = 0; i < n; i++) {

		//y = Euler(x, y, func_array, h);	   	// Calculate values with Euler or R_Cutt method
		y = R_Cutt(x, y, func_array, h);      


		for (int j = 0; j < N_EQS; j++) {  

			file_array[j] << y[j] << endl;   // Recording values to text file
		}
	}
}