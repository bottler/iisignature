#ifndef ROTATIONAL_INVARIANTS_HPP
#define ROTATIONAL_INVARIANTS_HPP

//This does the basic identification of raw linear rotational invariants in the signature
//as explained in the paper 
//"Rotation invariants of two dimensional curves based on iterated integrals"
//J Diehl 2013

#include<algorithm>
#include<utility>
#include<vector>
#include<cstdint>
//#include<bitset>

namespace RotationalInvariants {
  using std::vector;
  using std::pair;

  vector<vector<unsigned char>> possibleHalves(int n) {
    vector<unsigned char> base(n + n, 0);
    std::fill(base.begin() + n, base.end(), (unsigned char) 1);
    vector<vector<unsigned char>> out{ base };
    while (std::next_permutation(base.begin(), base.end())) {
      if (base[0] == 1)
        break;
      out.push_back(base);
    }
    return out;
  }

  //an Idx represents a word in two variables as a binary number
  //- and therefore as an offset into a level in the signature.
  // (If you set start1, it will be preceeded by a 1, so e.g. "112" 
  // and "12" will be represented as 1001b and 101b respectively
  // instead of both being 1, which may be easier to keep track of.
  // The rest of the code would have to change though.)
  using Idx = std::uint64_t;
  using Invariant = vector<pair<Idx, double>>;

  //expression being {1,0,0,1} means (x+iy)(x-iy)(x-iy)(x+iy)
  //this returns the real and imaginary parts 
  pair<Invariant, Invariant> multiplyOutTerm(vector<unsigned char>& expression)
  {
    bool start1 = false;
    Invariant real, imag;
    real.reserve(((size_t) 1u) << expression.size());
    imag.reserve(((size_t) 1u) << expression.size());
    real.emplace_back( start1 ? 1 : 0, 1 );
    for (unsigned char c : expression)
    {
      auto realHalf = real.insert(real.end(),imag.begin(), imag.end());
      auto imagHalf = imag.insert(imag.end(),real.begin(), realHalf);
      std::for_each(real.begin(), realHalf, [](pair<Idx, double>& p) {
        p.first *= 2;
      });
      std::for_each(imag.begin(), imagHalf, [](pair<Idx, double>& p) {
        p.first *= 2;
      });
      std::for_each(realHalf, real.end(), [c](pair<Idx, double>& p) {
        p.first = p.first * 2 + 1;
        if (!c)
          p.second *= -1;
      });
      std::for_each(imagHalf, imag.end(), [c](pair<Idx, double>& p) {
        p.first = p.first * 2 + 1;
        if (c)
          p.second *= -1;
      });
    }
    return { real,imag };
  }

  vector<Invariant> getInvariants(int n) {
    auto halves = possibleHalves(n);
    vector<Invariant> all;
    all.reserve(2 * halves.size());
    for (auto& v : halves) {
      auto p = multiplyOutTerm(v);
      all.push_back(std::move(p.first));
      all.push_back(std::move(p.second));
    }
    for (auto &invariant : all)
      std::sort(invariant.begin(), invariant.end());
    return all;
  }

#if 0
  template<size_t digits>
  void printOneTerm(Invariant p) {
    for (auto pp : p)
      std::cout << std::bitset<digits>(pp.first) << "," << pp.second << "  ";
    std::cout << "\n";
  }

  void print1() {
    vector<unsigned char> v{ 1,0,1,0 };
    auto p = multiplyOutTerm(v);
    printOneTerm<4>(p.first);
    printOneTerm<4>(p.second);
  }

  void print2() {
    for (auto& p : getInvariants(2))
      printOneTerm<2>(p);
  }
  void printAsMatrix() {
    std::cout << "[";
    int level = 4;
    bool first1 = true;
    for (auto& p : getInvariants(level/2)) {
      vector<double> expanded(((size_t)1u) << level);
      for (auto& term : p)
        expanded[(size_t)(term.first)] = term.second;
      if (!first1) {
        std::cout << ",\n";
      }
      first1 = false;
      std::cout << "[";
      bool first2 = true;
      for (auto pp : expanded) {
        if (!first2) {
          std::cout << ",";
        }
        first2 = false;
        std::cout << pp;
      }
      std::cout << "]";
    }
    std::cout << "]\n";
  }
#endif

  class Prepared {
  public:
    vector<vector<Invariant>> m_invariants;
    int m_level;
    size_t m_length;

    Prepared(int level) 
    : m_level(level), m_length(0)
    {
      m_invariants.assign(level / 2, {});
      for (int lev = 2; lev <= level; lev += 2) {
        m_invariants[lev / 2 - 1] = getInvariants(lev / 2);
        m_length += m_invariants[lev / 2 - 1].size();
      }
    }
  };
}

#endif
