#include "pch.h"

class Vector3;

class Vector0 {

private:

	double x;
	double y;

public:
	void Show();
	void Assign(double, double);
	Vector0();
	Vector0(double);
	Vector0(double, double);
	~Vector0();
	friend void Check(const Vector0 &, const Vector3 &);
};

void Vector0::Show() {

	cout << "x = " << x << ", y = " << y << endl;
}

void Vector0::Assign(double xx, double yy) {

	x = xx;
	y = yy;

}

Vector0::Vector0(double xx, double yy)
{

	x = xx;
	y = yy;
	cout << "Object created" << endl;
}

Vector0::Vector0(double xx)
{
	x = xx;
	y = 0;
	cout << "Object created" << endl;
}

Vector0::Vector0()
{
	x = 1;
	y = 0;
	cout << "Object created" << endl;
}


Vector0::~Vector0()
{
	cout << "Object destroyed" << endl;
}

class Vector3 {

private:

	double x;
	double y;

public:
	void Show();
	void Assign(double, double);
	Vector3();
	Vector3(double);
	Vector3(double, double);
	~Vector3();
	friend void Check(const Vector0 &, const Vector3 &);
};

void Vector3::Show() {

	cout << "x = " << x << ", y = " << y << endl;
}

void Vector3::Assign(double xx, double yy) {

	x = xx;
	y = yy;

}

Vector3::Vector3(double xx, double yy)
{

	x = xx;
	y = yy;
	cout << "Object created" << endl;
}

Vector3::Vector3(double xx)
{
	x = xx;
	y = 0;
	cout << "Object created" << endl;
}

Vector3::Vector3()
{
	x = 1;
	y = 0;
	cout << "Object created" << endl;
}


Vector3::~Vector3()
{
	cout << "Object destroyed" << endl;
}

void Check(const Vector0 & v1, const Vector3 & v2) {

	

		cout << "Largest x " << (v1.x + v2.x + abs(v1.x - v2.x)) / 2 << endl;
		cout << "Largest y " << (v1.y + v2.y + abs(v1.y - v2.y)) / 2 << endl;

}


void maighjn() {

	Vector0 a(2, 3);
	Vector3 b(5, 6);
	Check(a, b);
}