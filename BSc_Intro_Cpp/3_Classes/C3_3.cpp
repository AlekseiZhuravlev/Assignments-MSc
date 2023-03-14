#include "pch.h"


class Vector1 {

private:

	double x;
	double y;

public:
	void Show();
	void Assign(double, double);
	Vector1();
	Vector1(double);
	Vector1(double, double);
	~Vector1();
	Vector1 Expand(Vector1, int);
	Vector1 Conc(Vector1, Vector1);
	double Module();
	void Add(Vector1);
};

void Vector1::Show() {

	cout << "x = " << x << ", y = " << y << endl;
}

void Vector1::Assign(double xx, double yy) {

	x = xx;
	y = yy;
}

Vector1::Vector1(double xx, double yy)
{
	x = xx;
	y = yy;
	cout << "Object created" << endl;
}

Vector1::Vector1(double xx)
{
	x = xx;
	y = 0;
	cout << "Object created" << endl;
}

Vector1::Vector1()
{
	x = 1;
	y = 0;
	//cout << "Object created" << endl;
}

Vector1::~Vector1()
{
	//cout << "Object destroyed" << endl;
}

Vector1 Vector1::Expand(Vector1 a, int num) {

	Vector1 b;
	b.x = a.x*num;
	b.y = a.y*num;
	return b;
}

Vector1 Vector1::Conc(Vector1 a, Vector1 b) {

	Vector1 c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}


double Vector1::Module() {

	return sqrt(x*x + y*y);
}

void Vector1::Add(Vector1 a) {

	x += a.x;
	y += a.y;
}


void maiasdn() {
	srand(time(0));
	Vector1 v1(0,0), v2;
	double size = 1;
	double angle = 0;
	int i = 0;
	int results[30] = { 0 };

	for (int j = 0; j < 30; j++) {

		i = 0;
		v1.Assign(0, 0);
				while (v1.Module() < (double)50) {

					angle = (rand() % 361) / 57.3;
					v2.Assign(size * cos(angle), size * sin(angle));
					v1.Add(v2);
					i++;

				}
		cout << i << endl;
		results[j] = i;
	}

	double average = 0;
	double error = 0;

	for (int j = 0; j < 30; j++) {

		average += results[j];
		
	}

	average /= 30;

	for (int j = 0; j < 30; j++) {

		error += ((double)results[j] - average) * ((double)results[j] - average);

	}

	error = sqrt(error/30);

	cout << "Average " << average << endl;
	cout << "Error " << error << endl;

}