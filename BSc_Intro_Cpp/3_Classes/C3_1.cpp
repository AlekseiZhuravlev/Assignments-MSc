#include "pch.h"

class Vector3;

class Vector {

private:
		
	double x;
	double y;

public:
	void Show();
	void Assign(double, double);
	Vector();
	Vector(double);
	Vector(double, double);
	~Vector();
	Vector Expand(Vector, int);
	Vector Conc(Vector, Vector);
	double Module(Vector);

};

void Vector::Show() {

	cout << "x = " << x << ", y = " << y << endl;
}

void Vector::Assign(double xx, double yy) {

	x = xx;
	y = yy;

}

Vector::Vector(double xx, double yy)
{
	x = xx;
	y = yy;
	cout << "Object created" << endl;
}

Vector::Vector(double xx)
{
	x = xx;
	y = 0;
	cout << "Object created" << endl;
}

Vector::Vector()
{
	x = 1;
	y = 0;
	//cout << "Object created" << endl;
}

Vector::~Vector()
{
	//cout << "Object destroyed" << endl;
}

Vector Vector::Expand(Vector a, int num) {

	Vector b;
	b.x = a.x*num;
	b.y = a.y*num;
	return b;
}

Vector Vector::Conc(Vector a, Vector b) {

	Vector c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}


double Vector::Module(Vector a) {

	return sqrt(a.x*a.x + a.y*a.y);
}
void main() {

	Vector v1, v2;
	cout << "Enter x and y" << endl;
	double a, b;
	cin >> a;
	cin >> b;
	v1.Assign(a, b);
	v1.Show();
	v2.Show();

	cout << "Swapping" << endl;

	Vector v3 = v1;
	v1 = v2;
	v2 = v3;
	v1.Show();
	v2.Show();

	cout << "Expading" << endl;

	v1 = v1.Expand(v1, 3);
	v1.Show();

	cout << "Concantinating" << endl;
	v3 = v3.Conc(v1, v2);
	v3.Show();
;}