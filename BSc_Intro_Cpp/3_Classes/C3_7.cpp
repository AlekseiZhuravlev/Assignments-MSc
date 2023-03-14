#include "pch.h"



class Vector5 {

private:

	double x;
	double y;

public:
	Vector5 operator*(double) const;
	void Show();
	void Assign(double, double);
	Vector5();
	Vector5(double);
	Vector5(double, double);
	~Vector5();
	Vector5 Expand(Vector5, int);
	Vector5 Conc(Vector5, Vector5);
	double Module(Vector5);
	friend Vector5 operator*(const double, const Vector5);
};

void Vector5::Show() {

	cout << "x = " << x << ", y = " << y << endl;
}

void Vector5::Assign(double xx, double yy) {

	x = xx;
	y = yy;

}

Vector5::Vector5(double xx, double yy)
{
	x = xx;
	y = yy;
	cout << "Object created" << endl;
}

Vector5::Vector5(double xx)
{
	x = xx;
	y = 0;
	cout << "Object created" << endl;
}

Vector5::Vector5()
{
	x = 1;
	y = 0;
	//cout << "Object created" << endl;
}

Vector5::~Vector5()
{
	//cout << "Object destroyed" << endl;
}

Vector5 Vector5::Expand(Vector5 a, int num) {

	Vector5 b;
	b.x = a.x*num;
	b.y = a.y*num;
	return b;
}

Vector5 Vector5::Conc(Vector5 a, Vector5 b) {

	Vector5 c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}


double Vector5::Module(Vector5 a) {

	return sqrt(a.x*a.x + a.y*a.y);
}

Vector5 operator*(const double a, const Vector5 v1)
{
	return Vector5(v1.x*a, v1.y*a);
}
class Vector5;


Vector5 Vector5::operator*(double a) const
{
	return Vector5(a*x, a*y);
}

void mainsddsd() {

	double a = 5;
	Vector5 b(3, 3);
	b = b*a;
	b.Show();
	b = a * b;
	b.Show();
}