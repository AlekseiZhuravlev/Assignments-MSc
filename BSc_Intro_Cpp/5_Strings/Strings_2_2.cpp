#include "pch.h"

void Read() {

	ifstream file("Derivative.txt");
	ofstream file1("Derivative1.txt");

	char buff[50];

	file >> buff;
	while (buff[0] != NULL) {


		// Read first 4 words
		file >> buff;

		file >> buff;					// Read x
		file1 << buff << " ";

		file >> buff;
		file >> buff;

		file >> buff;					// Read f(x)
		file1 << buff << " ";

		file >> buff;
		file >> buff;

		file >> buff;					// Read f(x)
		file1 << buff << endl;

		file >> buff;
	}

}
