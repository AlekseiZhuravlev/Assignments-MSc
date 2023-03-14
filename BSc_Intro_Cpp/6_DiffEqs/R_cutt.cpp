#include "pch.h"
#define N_EQS 3


double* R_Cutt(double x, double* y, double(*func_array[N_EQS])(double, double), double h) {

	
	double k1, k2, k3, k4;

	for (int i = 0; i < N_EQS; i++) {

		k1 = func_array[i](x, y[i]);
		k2 = func_array[i](x + 0.5 * h, y[i] + 0.5 * h * k1);
		k3 = func_array[i](x + 0.5 * h, y[i] + 0.5 * h * k2);
		k4 = func_array[i](x + h, y[i] + h * k3);

		y[i] += (h/6) * (k1 + 2 * k2 + 2 * k3 + k4); // вычисление yi
	}

	return y;
}