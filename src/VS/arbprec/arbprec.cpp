#include<iostream>
#include<thread>
#include<random>
#include<type_traits>
#include<vector>
#include <boost/multiprecision/cpp_int.hpp>
#include"printtable.h"
#include "calcSignature.hpp"

namespace ExactSignature {
  using std::vector;

  typedef boost::multiprecision::cpp_rational Number;

  class Signature {
  public:
    vector<vector<Number>> m_data;

    template<typename Numeric>
    void sigOfSegment(int d, int m, const Numeric* segment) {
      m_data.resize(m);
      auto& first = m_data[0];
      first.resize(d);
      for (int i = 0; i<d; ++i)
        first[i] = (Number)segment[i];
      for (int level = 2; level <= m; ++level) {
        const auto& last = m_data[level - 2];
        auto& s = m_data[level - 1];
        s.assign(calcSigLevelLength(d, level), 0);
        int i = 0;
        for (auto l : last)
          for (auto p = segment; p<segment + d; ++p)
            s[i++] = (Number)(*p * l * Number(1, level));
      }
    }

    void sigOfNothing(int d, int m) {
      m_data.resize(m);
      m_data[0].assign(d, 0);
      for (int level = 2; level <= m; ++level) {
        auto& s = m_data[level - 1];
        s.assign(calcSigLevelLength(d, level), 0);
      }
    }

    //if a is the signature of path A, b of B, then
    //a.concatenateWith(d,m,b) makes a be the signature of the concatenated path AB
    //This is also the (concatenation) product of the elements a and b in the tensor algebra.
    void concatenateWith(int d, int m, const Signature& other) {
      for (int level = m; level>0; --level) {
        for (int mylevel = level - 1; mylevel>0; --mylevel) {
          int otherlevel = level - mylevel;
          auto& oth = other.m_data[otherlevel - 1];
          for (auto dest = m_data[level - 1].begin(),
            my = m_data[mylevel - 1].begin(),
            myE = m_data[mylevel - 1].end(); my != myE; ++my) {
            for (const Number& d : oth) {
              *(dest++) += d * *my;
            }
          }
        }
        auto source = other.m_data[level - 1].begin();
        for (auto dest = m_data[level - 1].begin(),
          e = m_data[level - 1].end();
          dest != e;)
          *(dest++) += *(source++);

      }
    }
    void swap(Signature& other) {
      m_data.swap(other.m_data);
    }
    void multiplyByConstant(Number c) {
      for (auto& a : m_data)
        for (auto& b : a)
          b *= c;
    }
    template<typename Numeric>
    void writeOut(Numeric* dest) const {
      for (auto& a : m_data)
        for (auto& b : a)
          *(dest++) = (Numeric)b;
          //*(dest++) = (Numeric)denominator(b);
    }
  };

  //This also calculates the concatenation product in the tensor algebra, 
  //but in the case where we assume 0 instead of 1 in the zeroth level.
  //It is not in-place
  Signature concatenateWith_zeroFirstLevel(int d, int m,
    const Signature& a,
    const Signature& b) {
    Signature out;
    out.sigOfNothing(d, m);
    for (int level = m; level>0; --level) {
      for (int alevel = level - 1; alevel>0; --alevel) {
        int blevel = level - alevel;
        auto& aa = a.m_data[alevel - 1];
        auto& bb = b.m_data[blevel - 1];
        auto dest = out.m_data[level - 1].begin();
        for (const Number& c : aa) {
          for (const Number& d : bb) {
            *(dest++) += d * c;
          }
        }
      }
    }
    return out;
  }

  void logTensorHorner(Signature& x) {
    const int m = (int)x.m_data.size();
    const int d = (int)x.m_data[0].size();
    if (m <= 1)
      return;
    Signature s, t;
    s.sigOfNothing(d, m - 1);
    t.sigOfNothing(d, m);
    for (int depth = m; depth > 0; --depth) {
      Number constant = (Number) 1.0 / depth;
      //make t be x*s up to level (1+m-depth). [this does nothing the first time round]
      for (int lev = 2; lev <= 1 + m - depth; ++lev) {
        //t.m_data[lev - 1] = x.m_data[lev - 1];
        auto& tt = t.m_data[lev - 1];
        std::fill(tt.begin(), tt.end(), (Number) 0.0);
        for (int leftLev = 1; leftLev < lev; ++leftLev) {
          int rightLev = lev - leftLev;
          auto& aa = x.m_data[leftLev - 1];
          auto& bb = s.m_data[rightLev - 1];
          auto dest = t.m_data[lev - 1].begin();
          for (const Number& c : aa)
            for (const Number& dd : bb)
              *(dest++) += dd * c;
        }
      }
      //make s be x*constant-t up to level (1+m-depth)
      if (depth>1)
        for (int lev = 1; lev <= 1 + m - depth; ++lev) {
          auto is = s.m_data[lev - 1].begin();
          auto ix = x.m_data[lev - 1].begin();
          auto es = s.m_data[lev - 1].end();
          auto it = t.m_data[lev - 1].begin();
          for (; is != es; ++is, ++it, ++ix)
            *is = constant * *ix - *it;
        }
    }
    //x isn't modified until this next bit.
    //make x be x-t
    for (int lev = 2; lev <= m; ++lev) {
      auto it = t.m_data[lev - 1].begin();
      auto ix = x.m_data[lev - 1].begin();
      auto ex = x.m_data[lev - 1].end();
      for (; ix != ex; ++ix, ++it)
        *ix -= *it;
    }
  }

}

std::mt19937 g_rnd;
using PathReal = double;
std::vector<PathReal> randomPath(int pathLength, int d){
  std::vector<PathReal> out(pathLength*d);
  std::uniform_real_distribution<PathReal> urd(0, 1);
  for (auto& d : out)
    d = urd(g_rnd);
  return out;
}

template<bool arbprec>
using SignatureType = std::conditional_t<arbprec,
  ExactSignature::Signature,
  CalcSignature::Signature>;

//the stroke ending at from ... the stroke ending before to
template<bool arbprec>
SignatureType<arbprec> sigPiece(const std::vector<PathReal>& path, int d, int m, size_t from, size_t to) {
  SignatureType<arbprec> s1, s2;
  std::vector<PathReal> displacement(d);
  for (size_t i = from; i < to; ++i) {
    for (int j = 0; j < d; ++j)
      displacement[j] = path[i*d + j] - path[(i - 1)*d + j];
    s1.sigOfSegment(d, m, displacement.data());
    if (i == from) {
      s2.swap(s1);
    }
    else {
      s2.concatenateWith(d, m, s1);
    }
  }
  return s2;
}

//Calculate the signature or log signature of path,
//using float or arbitrary precision
template<bool arbprec>
std::vector<double> sig(const std::vector<PathReal>& path, int d, int m, bool log) {
  size_t pathLength = path.size() / d;
  auto s2 = sigPiece<arbprec>(path, d, m, 1, pathLength);
  if (log)
    logTensorHorner(s2);
  std::vector<double> out(calcSigTotalLength(d, m));
  s2.writeOut(out.data());
  return out;
}


//like sig but split the work between processors
template<bool arbprec>
std::vector<double> sigParallel(const std::vector<PathReal>& path, int d, int m, bool log) {
  size_t pathLength = path.size() / d;
  size_t nThreads = 4;
  size_t each = (pathLength - 1) / nThreads;
  if (each < 1)
    return sig<arbprec>(path, d, m, log);
  size_t start = 1;
  std::vector<SignatureType<arbprec>> pieces(nThreads);
  std::vector<std::thread> threads;
  for (size_t thread = 0; thread < nThreads; ++thread) {
    size_t start = thread*each + 1;
    size_t end = (thread + 1 == nThreads) ? pathLength : (thread*each + each + 1);
    threads.emplace_back([&pieces,path,d,m,start, end,thread]() {
      pieces[thread] = sigPiece<arbprec>(path, d, m, start, end);
    });
  }
  for (auto& t : threads)
    t.join();
  auto s2 = pieces[0];
  for (size_t i = 1; i < nThreads; ++i)
    s2.concatenateWith(d, m, pieces[i]);
  if (log)
    logTensorHorner(s2);
  std::vector<double> out(calcSigTotalLength(d, m));
  s2.writeOut(out.data());
  return out;
}

void test() {
  using N = boost::multiprecision::cpp_rational;
  N a = 0.0625;
  N b = a - N(4, 5);
  std::cout << b << " "<<(double) b << "\n";
}


void go() {
  auto d = 2, m = 4, pathLength=4;
  auto path = randomPath(pathLength, d);
  bool log = true;
  auto s1 = sig<false>(path, d, m, log);
  //auto s2 = sig<true>(path, d, m, log);
  auto s2 = sigParallel<true>(path, d, m, log);

  //for (size_t i = 0; i < s1.size(); ++i) {
  //  std::cout << s1[i] << "  " << s2[i] << "\n";
  //}
  auto diffs = diffVectors(s1, s2);
  PrintTable::printTable(s1, s2, diffs);
  std::cout << "l2 norm " << l2norm(s1) << " "<< l2norm(s2) << "\n";
  std::cout << "l2 diff " << l2norm(diffs) << "\n";
  std::cout << "total diff "<<l1norm(diffs) << "\n";
  printMaxDiff(s1, s2);
  printMaxRelDiff(s1, s2);
  printGreatestWith0b(s1, s2);
}

int main() {
  //test();
  go();
}