#include "pch.h"

void hmain() {

	struct rec_ { int i; char cp[80]; };

	int n;
	cout << "Enter int." << endl;
	cin >> n;
	cin.get();

	rec_* arr = new rec_[n];

	for (int i = 0; i < n; i++) {

		arr[i].i = i;
		cout << "Enter string." << endl;
		cin.getline(arr[i].cp, 80);
	}

	cout << endl;

	for (int i = 1; i <= n; i++) {

		cout << arr[n-i].cp << endl;
	}

	delete[] arr;
}