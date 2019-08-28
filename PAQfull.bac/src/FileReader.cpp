/* 
 * File:   FileReader.cpp
 * Author: Andrew McMurdie
 * 
 * Created on April 1, 2013, 9:25 PM
 */

#include "FileReader.h"
#include "Samples.h"
#include <iostream>
#include <fstream>
#include <exception>
#include <cstdlib>

using namespace std;
using namespace PAQ_SOQPSK;

namespace PAQ_SOQPSK{
	FileReader::FileReader() {
	}

	FileReader::~FileReader() {
	}
	
	/**
	 * Loads the file into a Samples object, containing samples of the floats
	 * @param fileName
	 * @return A built Samples object, containing the I and Q samples
	 */
	Samples& FileReader::loadSamplesFile(string fileName) {
		Samples* inSamples;
		
		//Get the loader object, and load the file
		ifstream fileReader;
		fileReader.open(fileName.c_str());
		
		if(fileReader.is_open()) {
			cout << "Success loading file: " << fileName << endl;

			//Get the number of samples to load.
			string line;
			getline(fileReader, line);
			int numSamples = atoi(line.c_str());
			cout << "Number of samples is: " << numSamples << endl;

			//Load I Samples
			vector<float> I_Samples;
			for(int i=0; i<numSamples; i++) {
				getline(fileReader, line);
				I_Samples.push_back(atof(line.c_str()));
			}

			//Load Q Samples
			vector<float> Q_Samples;
			for(int i=0; i<numSamples; i++) {
				getline(fileReader, line);
				Q_Samples.push_back(atof(line.c_str()));
			}

			//Build Samples object
			inSamples = new Samples();
			inSamples->setI(I_Samples);
			inSamples->setQ(Q_Samples);
		} else {
			//Error loading file. Simply throw the exception to exit
			cout << "Error loading file: " << fileName << endl;
			throw exception();
		}

		fileReader.close();

		return *inSamples;
	}

	/**
	 * Loads the DetectionFilter from the Matlab file. Returns it as a vector of
	 * floats. The default detectionFilter is 20 elements long.
	 * @param fileName
	 * @return
	 */
	vector<float>& FileReader::loadDetectionFilter(string fileName) {
		vector<float>* detectionFilter;
		//Get the loader object, and load the file
		ifstream fileReader;
		fileReader.open(fileName.c_str());

		if(fileReader.is_open()) {
			cout << "Detection Filter file loaded successfully" << endl;
			
			detectionFilter = new vector<float>;
			string line;
			int i=0;
			while (fileReader.good()) {
				getline(fileReader, line);
				float filterTap = atof(line.c_str());
				//cout << "Filter tap[" << i << "]: " << filterTap << endl;
				i++;
				detectionFilter->push_back(filterTap);
			}
			
		} else {
			cout << "Error loading Detection Filter file: " << fileName << endl;
			throw exception();
		}
		
		fileReader.close();
		return *detectionFilter;
	}

	/**
	 * Loads the file into a Samples object, containing samples of the floats
	 * @param fileName
	 * @return A built Samples object, containing the I and Q samples
	 */
	Samples& FileReader::loadCSVFile(string fileName) {
		Samples* inSamples;

		//Get the loader object, and load the file
		ifstream fileReader;
		fileReader.open(fileName.c_str());
		
		if(fileReader.is_open()) {
			//cout << "CSV File: '" << fileName << "' opened successfully" << endl;

			vector<float> I_Samples;
			vector<float> Q_Samples;

			unsigned int commaPosition;
			//int i=0;
			int lineIndex = 0;
			//while((fileReader.good()) && (lineIndex < 500)) {
			while(fileReader.good()) {
				//cout << "      On line: " << i << endl;
				//i++;

				//Pull the line, and then find the comma. Split the string into
				//the two pieces, and process accordingly.
				string line;
				getline(fileReader, line);
				if(line.length() == 0) break;

				commaPosition = line.find(',');
				if(commaPosition != string::npos) {
					string i_str = line.substr(0, commaPosition);
					string q_str = line.substr(commaPosition + 1, string::npos);
					float i_value = atof(i_str.c_str());
					float q_value = atof(q_str.c_str());

					I_Samples.push_back(i_value);
					Q_Samples.push_back(q_value);
				} else {
					//Simply terminate for now
					//throw exception();
					cout << "ERROR IN LOAD CSV FILE" << endl;
				}

				lineIndex++;
			}

			//cout << "Number of samples is: " << I_Samples.size() << endl;

			//Save these to the new Samples object
			inSamples = new Samples();
			inSamples->setI(I_Samples);
			inSamples->setQ(Q_Samples);

			//cout << "Assigned I and Q samples." << endl;
		} else {
			cout << "Could not open file '" << fileName << "'" << endl;
		}
		
		fileReader.close();
		return *inSamples;
	}
	
	/**
	 * Loads in the error samples.
	 */
	vector<float>& FileReader::loadErrorSamplesFile(string fileName) {
		vector<float>* errorSamples;
		//Get the loader object, and load the file
		ifstream fileReader;
		fileReader.open(fileName.c_str());

		if(fileReader.is_open()) {
			cout << "File opened successfully" << endl;

			errorSamples = new vector<float>;
			string line;
			while (fileReader.good()) {
				getline(fileReader, line);
				float filterTap = atof(line.c_str());

				errorSamples->push_back(filterTap);
			}

		} else {
			cout << "Error loading Detection Filter file: " << fileName << endl;
			throw exception();
		}

		fileReader.close();
		return *errorSamples;
	}

	/**
	 * Loads an S Vector and returns it as a float. The vector will be 256*256 elements long
	 * as a single dimensional array.
	 */
	vector<float>& FileReader::loadSVectorFile(string fileName) {
		vector<float>* sVector = new vector<float>;

		//Get the loader object, and load the file
		ifstream fileReader;
		fileReader.open(fileName.c_str());

		if(fileReader.is_open()) {
			cout << "File opened successfully" << endl;

			string line;
			while (fileReader.good()) {
				getline(fileReader, line);
				float sValue = atof(line.c_str());

				sVector->push_back(sValue);
			}

		} else {
			cout << "Error loading Detection Filter file: " << fileName << endl;
			throw exception();
		}

		fileReader.close();


		return *sVector;
	}

}
