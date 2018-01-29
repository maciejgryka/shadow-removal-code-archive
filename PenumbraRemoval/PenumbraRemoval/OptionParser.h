#ifndef OPTION_PARSER_H
#define OPTION_PARSER_H

#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <utility>
#include <sstream>

#include "Options.h"

class OptionParser {
public:
	OptionParser(const std::string& file);
	~OptionParser() {};

  Options options() const { return options_; }

private:
	Options options_;

	OptionParser();

	bool isComment(const std::string& line) {
		std::stringstream ss;
		std::string firstChar;
		ss << line[0];
		ss >> firstChar;
		return streq(firstChar, "#");
	};

	void parseLine(const std::string& line);

	bool streq(const std::string& s1, const std::string& s2) {
		return (s1.compare(s2) == 0);
	};

	std::vector<std::string> split(const std::string& str, const char& delim);

	// split the line at the first space, everything afterwards is the value
	std::pair<std::string, std::string> splitOptionLine(const std::string& str, const char& delim);
};

#endif