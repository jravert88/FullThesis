/* 
 * File:   FileReader.h
 * Author: Andrew McMurdie
 *
 * Created on April 1, 2013, 9:25 PM
 */

#ifndef FILEREADER_H
#define	FILEREADER_H

#include <vector>
#include <string>
#include "Samples.h"
using namespace std;
using namespace PAQ_SOQPSK;

namespace PAQ_SOQPSK{
	class FileReader {
	public:
		FileReader();
		virtual ~FileReader();


		Samples& loadSamplesFile(string fileName);
		Samples& loadCSVFile(string fileName);
		vector<float>& loadErrorSamplesFile(string fileName);
		vector<float>& loadDetectionFilter(string fileName);
		vector<float>& loadSVectorFile(string fileName);
	private:

	};

}

#endif	/* FILEREADER_H */

