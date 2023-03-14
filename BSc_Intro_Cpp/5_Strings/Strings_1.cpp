#include "pch.h"
#include <fstream>


void Average_Score() {


	setlocale(LC_ALL, "rus");

	ifstream file("scores.txt");

	char buff[200];
	int math = 0, phys = 0, chem = 0;


	file.getline(buff, 200);		// Read first 2 strings
	file.getline(buff, 200);

	file >> buff;
	while (buff[0] != NULL) {


							// Read first 4 words
		file >> buff;
		file >> buff;
		file >> buff;

		file >> buff;					// Read math mark
		phys += buff[0] - '0';

		file >> buff;					// Read math mark
		math += buff[0] - '0';

		file >> buff;					// Read math mark
		chem += buff[0] - '0';

		file >> buff;
	}

	cout << phys << endl;
	cout << math << endl;
	cout << chem << endl;
}