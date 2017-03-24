#ifndef PRINTTABLE_H
#define PRINTTABLE_H

#include<iostream>
#include<iomanip>

// printTable (a,b,c,..) prints a table to cout whose columns are a,b,c

namespace PrintTable {
  using std::ostream;
  template<class S>
  void checkSize(size_t size, const S& a) {
    if (a.size() != size)
      throw std::runtime_error("PrintTable: distinct sizes of input");
  }
  template<class S, class... T>
  void checkSize(size_t size, const S& a, const T&... b) {
    checkSize(size, a);
    checkSize(size, b...);
  }
  template<class S, class... T>
  size_t getCommonSize(const S& a, const T&... b) {
    size_t size = a.size();
    checkSize(size, b...);
    return size;
  }
  template<class S>
  void printPieces(size_t i, ostream& s, const S& a) {
    s << std::setw(9) << std::setprecision(4) << a[i];
  }
  template<class S, class... T>
  void printPieces(size_t i, ostream& s, const S& a, const T& ... b) {
    printPieces(i, s, a);
    s << " ";
    printPieces(i, s, b...);
  }
  template<class... T>
  void printRow(size_t i, ostream& s, const T&... a) {
    printPieces(i, s, a...);
    s << "\n";
  }

  template<class... T>
  void printTable(const T&... a) {
    size_t size = getCommonSize(a...);
    for (size_t i = 0; i < size; ++i) {
      printRow(i, std::cout, a...);
    }
  }
}

template<typename T> T diffVectors(const T& a, const T& b) {
  size_t size = PrintTable::getCommonSize(a, b);
  T out(size);
  for (size_t i = 0; i < size; ++i)
    out[i] = a[i] - b[i];
  return out;
}

double l2distance(const std::vector<double>& a, const std::vector<double>& b) {
  size_t size = PrintTable::getCommonSize(a, b);
  double o = 0;
  for (size_t i = 0; i < size; ++i) {
    double d = (a[i] - b[i]);
    o += d * d;
  }
  return sqrt(o);
}

double l2norm(const std::vector<double>& a) {
  double o = 0;
  for (double x : a) {
    o += x * x;
  }
  return sqrt(o);
}

double l1norm(const std::vector<double>& a) {
  double o = 0;
  for (double x : a) {
    o += std::abs(x);
  }
  return sqrt(o);
}

void printMaxDiff(const std::vector<double>& a, const std::vector<double>& b) {
  size_t size = PrintTable::getCommonSize(a, b);
  size_t found = 0;
  double foundDiff = std::abs(a[0] - b[0]);
  for (size_t i = 1; i < size; ++i) {
    double diff = std::abs(a[i] - b[i]);
    if (diff > foundDiff) {
      foundDiff = diff;
      found = i;
    }
  }
  std::cout << "greatest diff " << a[found] << " " << b[found] << " " << foundDiff << "\n";
}
void printMaxRelDiff(const std::vector<double>& a, const std::vector<double>& b) {
  size_t size = PrintTable::getCommonSize(a, b);
  size_t found = 0;
  double foundDiff = std::abs(a[0] / b[0]-1);
  for (size_t i = 1; i < size; ++i) {
    if (b[i] != 0) {
      double diff = std::abs(a[i] / b[i]-1);
      if (diff > foundDiff) {
        foundDiff = diff;
        found = i;
      }
    }
  }
  std::cout << "greatest relative diff " << a[found] << " " << b[found] << " " << foundDiff << "\n";
}
void printGreatestWith0b(const std::vector<double>& a, const std::vector<double>& b) {
  size_t size = PrintTable::getCommonSize(a, b);
  double o = 0;
  for (size_t i = 0; i < size; ++i) {
    if (b[i] == 0 && std::abs(a[i]) > o)
      o = std::abs(a[i]);
  }
  std::cout << "greatest where right answer 0: " << o << "\n";
}


#endif