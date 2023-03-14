#include "pch.h"


class Vector2 {

private:

	double x;
	double y;

public:
	void Show();
	void Assign(double, double);
	Vector2();
	Vector2(double);
	Vector2(double, double);
	Vector2(Vector2, Vector2);
	~Vector2();
	Vector2 Expand(Vector2, int);
	Vector2 Conc(Vector2, Vector2);
	double Module();
	void Add(Vector2);
};

void Vector2::Show() {

	cout << "x = " << x << ", y = " << y << endl;
}

void Vector2::Assign(double xx, double yy) {

	x = xx;
	y = yy;

}

Vector2::Vector2(double xx, double yy)
{
	Vector2 s;
	x = s.x + xx;
	y = s.y + yy;
	cout << "Object created" << endl;
}

Vector2::Vector2(double xx)
{
	x = xx;
	y = 0;
	cout << "Object created" << endl;
}

Vector2::Vector2()
{
	x = 1;
	y = 0;
	cout << "Object created" << endl;
}

Vector2::Vector2(Vector2 a, Vector2 b)
{
	Vector2 s;
	x = s.x + a.x + b.x;
	y = s.y + a.y + b.y;
	//cout << "Object created" << endl;
}

Vector2::~Vector2()
{
	cout << "Object destroyed" << endl;
}

Vector2 Vector2::Expand(Vector2 a, int num) {

	Vector2 b;
	b.x = a.x*num;
	b.y = a.y*num;
	return b;
}

Vector2 Vector2::Conc(Vector2 a, Vector2 b) {

	Vector2 c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}


double Vector2::Module() {

	return sqrt(x*x + y * y);
}

void Vector2::Add(Vector2 a) {

	x += a.x;
	y += a.y;
}



void mainnmnm() {
	
	Vector2 v[3] = { Vector2(2., 2.), Vector2(3., 3.) };
	for (int i = 0; i < 3; i++) {

		v[i].Show();
	}
}