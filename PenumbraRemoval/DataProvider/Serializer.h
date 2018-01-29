#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <iostream>
#include <vector>

#include <Eigen/Core>

#include "Types.h"

typedef unsigned char PixelData;

namespace Serializer {

template<typename T>
std::vector<T> StringToStdVector(const std::string& line, int n_elements = 0) {
  // if number of elements was not specified, count the commas
  if (n_elements == 0) { n_elements = CountCommas(line) + 1; }
  // fill in the vector
  std::vector<T> vec(n_elements);
  std::stringstream ss(line);
	std::string value;
	for (int el = 0; el < n_elements; ++el) {
		getline(ss, value, ',');
		istringstream(value) >> vec[el];
	}
	return vec;
}

template<typename T>
static std::string StdVectorToString(const std::vector<T>& vec) {
  std::ostringstream oss;
  std::vector<T>::const_iterator cit = vec.begin();
  oss << *cit << ",";
  for (; cit != vec.end(); ++cit) {
    oss << *cit << ",";
  }
  return oss.str();
}

} // namespace Serializer

class StreamWriter {
public:
  StreamWriter(std::ostream* out_stream): out_stream_(out_stream) {}
  ~StreamWriter();
  
  void WritePixel(PixelData val) { WriteStream<PixelData>(val, *out_stream_); }
  void WriteInt(int val) { WriteStream<int>(val, *out_stream_); }
  void WriteFloat(float val) { WriteStream<float>(val, *out_stream_); }

  void WritePixelMatrix(const EigenMat& mat);
  void WriteFloatMatrix(const EigenMat& mat);
  
  template <typename T>
  static void WriteStream(const T& val, std::ostream& out_stream) {
    out_stream.write((char*)(&val), sizeof(T));
  }

  template <typename T>
  void WriteVector(const std::vector<T>& vec);

  void WriteIntVector(const std::vector<int>& vec);
  void WriteFloatVector(const std::vector<float>& vec);

private:
  std::ostream* out_stream_;

  StreamWriter() {}
  StreamWriter(const StreamWriter&) {}
};

class StreamReader {
public:
  StreamReader(std::istream* in_stream): in_stream_(in_stream) {}
  ~StreamReader();

  int ReadPixel() { return ReadStream<PixelData>(*in_stream_); }
  int ReadInt() { return ReadStream<int>(*in_stream_); }
  float ReadFloat() { return ReadStream<float>(*in_stream_); }

  EigenMat ReadPixelMatrix();
  EigenMat ReadFloatMatrix();

  std::vector<int> ReadIntVector();
  std::vector<float> ReadFloatVector();

  template <typename T>
  static T ReadStream(std::istream& in_stream) {
    T val;
    in_stream.read((char*)(&val), sizeof(T));
    return val;
  }

  template <typename T>
  std::vector<T> ReadVector();

private:
  std::istream* in_stream_;

  StreamReader();
  StreamReader(const StreamReader&);
};

class MatrixSerializer {
};

#endif // SERIALIZER_H