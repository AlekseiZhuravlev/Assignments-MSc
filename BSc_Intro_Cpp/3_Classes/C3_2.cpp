#include "pch.h"


class MyTime {

	private:
		int h;
		int m;

	public:
		MyTime();
		MyTime(int);
		MyTime(int, int);
		void Set(int, int);
		void Show();
		MyTime Interval(MyTime, MyTime);
};

MyTime::MyTime() {

	h = 0;
	m = 0;
}

MyTime::MyTime(int a) {

	h = (int)(a / 60);
	m = a % 60;
}

MyTime::MyTime(int a, int b) {

	h = a + (int)(b / 60);
	m = b % 60;
}

void MyTime::Show() {

	cout << h << ":" << m << endl;
}

void MyTime::Set(int a, int b) {

	h = a + (int)(b / 60);
	m = b % 60;
}


MyTime MyTime::Interval(MyTime x, MyTime y) {

	MyTime a;
	a.h = x.h - y.h;
	a.m = x.m - y.m;
	return a;
}
void maisdfn() {

	MyTime v1, v2, v3;
	cout << "Enter h and m" << endl;
	int a, b;
	cin >> a;
	cin >> b;
	v1.Set(a, b);
	v2.Set(12, 65);
	v1.Show();
	v2.Show();


	v3 = v3.Interval(v1, v2);
	v3.Show();
}