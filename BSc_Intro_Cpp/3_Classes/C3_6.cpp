#include "pch.h"



class Vector4 {

private:

	double x;
	double y;

public:
	Vector4 operator-() const;
	void Show();
	void Assign(double, double);
	Vector4();
	Vector4(double);
	Vector4(double, double);
	~Vector4();
	Vector4 Expand(Vector4, int);
	Vector4 Conc(Vector4, Vector4);
	double Module(Vector4);
	friend Vector4 operator-(const Vector4, const Vector4);	
};

void Vector4::Show() {

	cout << "x = " << x << ", y = " << y << endl;
}

void Vector4::Assign(double xx, double yy) {

	x = xx;
	y = yy;

}

Vector4::Vector4(double xx, double yy)
{
	x = xx;
	y = yy;
	cout << "Object created" << endl;
}

Vector4::Vector4(double xx)
{
	x = xx;
	y = 0;
	cout << "Object created" << endl;
}

Vector4::Vector4()
{
	x = 1;
	y = 0;
	//cout << "Object created" << endl;
}

Vector4::~Vector4()
{
	//cout << "Object destroyed" << endl;
}

Vector4 Vector4::Expand(Vector4 a, int num) {

	Vector4 b;
	b.x = a.x*num;
	b.y = a.y*num;
	return b;
}

Vector4 Vector4::Conc(Vector4 a, Vector4 b) {

	Vector4 c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}


double Vector4::Module(Vector4 a) {

	return sqrt(a.x*a.x + a.y*a.y);
}

Vector4 operator-(const Vector4 v1, const Vector4 v2)
{
	return Vector4(v1.x - v2.x, v1.y - v2.y);
}
class Vector4;


Vector4 Vector4::operator-() const
{
	return Vector4(-x, -y);
}

void mainsavvsa() {

	Vector4 a(2, 3);
	Vector4 b(3, 3);
	b = -b;
	b.Show();
	b = a - b;
	b.Show();
}