#include "Serializer.h"

StreamWriter::~StreamWriter() {}

void StreamWriter::WritePixelMatrix(const EigenMat& data) {
  int rows = static_cast<int>(data.rows());
  int cols = static_cast<int>(data.cols());
  WriteInt(rows);
  WriteInt(cols);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      WritePixel(static_cast<PixelData>(data(r, c) * 255.f));
    }
  }
}

void StreamWriter::WriteFloatMatrix(const EigenMat& data) {
  int rows = static_cast<int>(data.rows());
  int cols = static_cast<int>(data.cols());
  WriteInt(rows);
  WriteInt(cols);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      WriteFloat(data(r, c));
    }
  }
}

template <typename T>
void StreamWriter::WriteVector(const std::vector<T>& vec) {
  int n_elements(static_cast<int>(vec.size()));
  WriteInt(n_elements);
  for (std::vector<T>::const_iterator it = vec.begin(); it != vec.end(); ++it) {
    WriteStream<T>(*it, *out_stream_);
  }
}

void StreamWriter::WriteIntVector(const std::vector<int>& vec) { 
  WriteVector<int>(vec);
}

void StreamWriter::WriteFloatVector(const std::vector<float>& vec) {
  WriteVector<float>(vec);
}

StreamReader::~StreamReader() {}

EigenMat StreamReader::ReadPixelMatrix() {
  int rows = ReadInt();
  int cols = ReadInt();

  EigenMat data(rows, cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      data(r, c) = static_cast<float>(ReadPixel()) / 255.f;
    }
  }
  return data;
}

EigenMat StreamReader::ReadFloatMatrix() {
  int rows = ReadInt();
  int cols = ReadInt();

  EigenMat data(rows, cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      data(r, c) = ReadFloat();
    }
  }
  return data;
}

template <typename T>
std::vector<T> StreamReader::ReadVector() {
  int n_elements = ReadInt();
  std::vector<T> vec(n_elements);
  for (std::vector<T>::iterator it = vec.begin(); it != vec.end(); ++it) {
    *it = ReadStream<T>(*in_stream_);
  }
  return vec;
}

std::vector<int> StreamReader::ReadIntVector() {
  return ReadVector<int>();
}

std::vector<float> StreamReader::ReadFloatVector() {
  return ReadVector<float>();
}