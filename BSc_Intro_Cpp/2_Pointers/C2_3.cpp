#include "pch.h"


template <typename T> 

T** Create(T h = NULL, int n = 10, T j = NULL) {


	T** arr = new T*[n];

	for (int i = 0; i < n; i++) {

		arr[i] = new T[i];

		for (int f = 0; f <= i; f++) {

			arr[i][f] = j;
		}
	}

	return arr;
}


template <typename T>

T** Show(T** arr, int length) {


	for (int i = 0; i < length; i++) {

		for (int j = 0; j <= i; j++) {

			cout << arr[i][j] << " ";
		}
		cout << endl;
	}

	return arr;
}

template <typename T>

T** Assign(T** arr, int length, T value, int i, int k) {

	if (i < length && k <=  i) {

		arr[i][k] = value;
	}
	else {

		cout << "Invalid adress." << endl;
	}
	return arr;
}

template <typename T>

T Read(T** arr, int length, int i, int k) {

	if (i < length && k <= i) {

		return arr[i][k];
	}
	else {

		cout << "Invalid adress." << endl;
		return NULL;
	}
}

void main() {

	double a = 2;
	double** as = Create(a);
	Show(as, 10);
	as = Assign(as, 10, (double)9, 2, 1);
	Show(as, 10);
	cout << endl << Read(as, 10, 1, 1) << endl << endl;

	int b = 2;
	int** bs = Create(b, 8);

	for (int i = 0; i < 8; i++) {

		for (int j = 0; j <= i; j++) {

			bs[i][j] = (i + 1) * 10 + (j + 1);
		}
	}
	Show(bs, 8);
}