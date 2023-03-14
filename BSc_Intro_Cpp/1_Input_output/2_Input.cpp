#include "pch.h"

int dmain() {


	int m;
	cout << "Enter int." << endl;
	cin >> m;
	cin.get();
	char* sentence = new char[m+1];

	cout << "Enter sentence." << endl;
	cin.getline(sentence, m);

	char a;
	for (int i = 0; i < int(strlen(sentence) / 2); i++) {

		a = sentence[i];
		sentence[i] = sentence[strlen(sentence) - i - 1];
		sentence[strlen(sentence) - i - 1] = a;
	}

	cout << endl << sentence << endl;

	
		delete sentence;

	exit(0);
}