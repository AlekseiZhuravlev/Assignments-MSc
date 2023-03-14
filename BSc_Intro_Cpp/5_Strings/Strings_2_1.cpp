#include "pch.h"

double f(double x)
{
	return x * x;
}

void Derivative ()
{
	double x, h, fl, fr, fc, f2;
	ofstream file("Derivative.txt");

	x = 1;
	h = 0.1;

	for (int i = 0; i < 20; i++) {

		fc = (f(x + h) - f(x)) / (h);

		file << "x = " << x;
		file << " f(x) = " << f(x);
		file << " f'(x) = " << fc << endl;

		x += h;
	}
}