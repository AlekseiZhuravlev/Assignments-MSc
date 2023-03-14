#include "pch.h"
#include <conio.h>
int smain() {

	char sentence[80];
	char sentence1[80];
	cout << "Enter sentence." << endl;
	cin.getline(sentence, 80);

	int m;
	cout << "Enter int." << endl;
	cin >> m;
	

	cout << endl << sentence << endl;
	cout << m << endl << endl;

	cout << "Press Enter ->" << endl << flush;
	cin.get();
	cin.get();

	
	exit(123);
}