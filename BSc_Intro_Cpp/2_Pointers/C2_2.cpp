#include "pch.h"

#define TOCOMPILE 1

#if TOCOMPILE==0

void Show(const char* word = "Nothing entered", int n = 1) {

	for (int i = 0; i < n; i++) {

		cout << word << endl;
	}
}
#else


void Show() {


	cout << "Nothing entered" << endl;
	
}

void Show(const char* word) {


	cout << word << endl;
}

void Show(const char* word, int n) {

	for (int i = 0; i < n; i++) {

		cout << word << endl;
	}
}
#endif

void twomain(){

	char *word = new char[80];

	cout << "Enter sentence." << endl;
	cin.getline(word, 80);

	cout << endl << "Empty"  << endl;
	Show();
	cout << endl << "One"  << endl;
	Show(word);
	cout << endl << "Five" << endl;
	Show(word, 5);
	delete[] word;
}