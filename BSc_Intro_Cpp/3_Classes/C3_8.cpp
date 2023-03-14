#include "pch.h"



class Vector7 {

private:

	double x;
	double y;

public:
	Vector7 operator-() const;
	
	void Show();
	void Assign(double, double);
	Vector7();
	Vector7(double);
	Vector7(double, double);
	~Vector7();
	Vector7 Expand(Vector7, int);
	Vector7 Conc(Vector7, Vector7);
	const double Module();
	friend Vector7 operator-(const Vector7, const Vector7);	friend Vector7 operator+(const Vector7, const Vector7);	friend bool operator<(const Vector7, const Vector7);
	friend bool operator>(const Vector7, const Vector7);
};

void Vector7::Show() {

	cout << "x = " << x << ", y = " << y << endl;
}

void Vector7::Assign(double xx, double yy) {

	x = xx;
	y = yy;

}

Vector7::Vector7(double xx, double yy)
{
	x = xx;
	y = yy;
	cout << "Object created" << endl;
}

Vector7::Vector7(double xx)
{
	x = xx;
	y = 0;
	cout << "Object created" << endl;
}

Vector7::Vector7()
{
	x = 1;
	y = 0;
	//cout << "Object created" << endl;
}

Vector7::~Vector7()
{
	//cout << "Object destroyed" << endl;
}

Vector7 Vector7::Expand(Vector7 a, int num) {

	Vector7 b;
	b.x = a.x*num;
	b.y = a.y*num;
	return b;
}

Vector7 Vector7::Conc(Vector7 a, Vector7 b) {

	Vector7 c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}


const double Vector7::Module() {

	return sqrt(x*x + y*y);
}

Vector7 operator-(const Vector7 v1, const Vector7 v2)
{
	return Vector7(v1.x - v2.x, v1.y - v2.y);
}
Vector7 operator+(const Vector7 v1, const Vector7 v2)
{
	return Vector7(v1.x + v2.x, v1.y + v2.y);
}

bool operator>(Vector7 v1, Vector7 v2)
{

	return v1.Module()>v2.Module();
}

bool operator<(Vector7 v1, Vector7 v2)
{

	return (v1.Module() < v2.Module());
}

class Vector7;


Vector7 Vector7::operator-() const
{
	return Vector7(-x, -y);
}

void mainvxxv() {

	Vector7 a(2, 3);
	Vector7 b(3, 3);
	b = a-b;
	b.Show();
	b = a + b;
	b.Show();
	cout << (a < b);
	cout << (a > b);
}