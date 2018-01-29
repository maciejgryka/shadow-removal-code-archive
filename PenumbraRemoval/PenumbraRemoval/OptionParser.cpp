#include "OptionParser.h"

using namespace std;

OptionParser::OptionParser(const string& file) {
	ifstream  infile;
	infile.open(file, ios::in);

	if (!infile.good()) throw;

	string line;
	while(!getline(infile,line).eof()) {
		parseLine(line);	
	}

	infile.close();
}

void OptionParser::parseLine(const string& line) {
	if (line.empty() || isComment(line)) {
		return;
	}

  options_.SetParam(splitOptionLine(line, ' '));
}

vector<string> OptionParser::split(const string& str, const char& delim) {
	stringstream iss;
	iss << str;
		
	vector<string> strings;
	string token;

	while (getline(iss, token, delim)) {
		strings.push_back(token);
	}
	return strings;
}

pair<string, string> OptionParser::splitOptionLine(const string& str, const char& delim) {
	// split the line
  vector<string> parts(split(str, delim));
	
  // set the key to be the first part
  pair<string, string> key_val;
	key_val.first = parts[0];
	
  // set the value to be the rest
	string value;
	for (int s = 1; s < parts.size(); ++s) {
		value += parts[s];
	}
	key_val.second = value;

	return key_val;
}