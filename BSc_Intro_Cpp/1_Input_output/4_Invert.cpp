#include "pch.h"

void main() {

	struct rec_ { int i; char *cp; };

	int n;
	cout << "Enter int." << endl;
	cin >> n;
	cin.get();
	rec_* arr = new rec_[n];
	char cont[100];

	for (int i = 0; i < n; i++) {


		arr[i].i = i;



		cout << "Enter string." << endl;
		cin.getline(cont, 100);
		
		arr[i].cp = new char[strlen(cont)+i+1];
		strcpy_s(arr[i].cp, strlen(cont)+1, cont);
		
	}

	int ch;
	cout << "Enter N to invert" << endl;
	cin >> ch;

	for (int i = 0; i < n; i++) {
		if (i == ch) {

			char a;
				for (int j = 0; j < int(strlen(arr[i].cp) / 2); j++) {

					a = arr[i].cp[j];
					arr[i].cp[j] = arr[i].cp[strlen(arr[i].cp) - j - 1];
					arr[i].cp[strlen(arr[i].cp) - j - 1] = a;
				}

			cout << arr[i].i << ": " << arr[i].cp << endl;
			delete[] arr[i].cp;
		}
		else {

			cout << arr[i].i << ": " << arr[i].cp << endl;
			delete[] arr[i].cp;
		}
	}

	delete[] arr;
}