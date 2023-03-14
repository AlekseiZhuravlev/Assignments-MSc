#include "pch.h"


struct Root { int h; int m; };

void ShowTime(const Root &clock){

	cout << "Time:" << endl;
	cout << clock.h << ":" << clock.m << endl;
}

Root DelayTime(const Root &t1, const Root &t2) {

	Root interval;
	Root& inter = interval;
	
	inter.h = abs(t1.h - t2.h);
	inter.m = abs(t1.m - t2.m) % 60;

	return inter;
}

void onemain() {


	Root r1;
	Root r2;

	const Root &r1r = r1;
	const Root &r2r = r2;
	cout << "Enter time1" << endl;
	cout << "Hours: ";
	cin >> r1.h;
	cout << "Minutes: ";
	cin >> r1.m;
	cout << endl;

	cout << "Enter time2" << endl;
	cout << "Hours: ";
	cin >> r2.h;
	cout << "Minutes: ";
	cin >> r2.m;
	cout << endl;

	ShowTime(r1r);
	

	ShowTime(DelayTime(r1r, r2r));
}