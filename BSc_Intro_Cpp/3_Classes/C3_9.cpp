#include "pch.h"


class MyTime1 {

private:
	int h;
	int m;

public:
	MyTime1();
	MyTime1(int);
	MyTime1(int, int);
	void Set(int, int);
	void Show();
	MyTime1 Interval(MyTime1, MyTime1);
	friend ostream & operator<<(ostream &, const MyTime1 &);
};

MyTime1::MyTime1() {

	h = 0;
	m = 0;
}

MyTime1::MyTime1(int a) {

	h = (int)(a / 60);
	m = a % 60;
}

MyTime1::MyTime1(int a, int b) {

	h = a + (int)(b / 60);
	m = b % 60;
}

void MyTime1::Show() {

	cout << h << ":" << m << endl;
}

void MyTime1::Set(int a, int b) {

	h = a + (int)(b / 60);
	m = b % 60;
}


MyTime1 MyTime1::Interval(MyTime1 x, MyTime1 y) {

	MyTime1 a;
	a.h = x.h - y.h;
	a.m = x.m - y.m;
	return a;
}

ostream & operator<<(ostream & os, const MyTime1 & v)
{
	os << v.h << ":" << v.m << endl;
	return os;
}

void jkkjmain() {

	MyTime1 v1, v2, v3;
	v1.Set(1, 56);
	v2.Set(12, 65);
	cout << v1;


}