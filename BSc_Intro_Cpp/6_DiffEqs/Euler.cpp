#include "pch.h"
#define N_EQS 3


double* Euler(double x, double* y, double(*func_array[N_EQS])(double, double), double h) {
	

	for (int i = 0; i < N_EQS; i++)	{
		y[i] += h * func_array[i](x, y[i]); // вычисление yi
	}

	return y;
}
