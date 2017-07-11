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
#include<bitset>
#include<string>

namespace RotationalInvariants {
  using std::vector;
  using std::pair;

  //Returns all the combinations of n 0s and n 1s which begin with 0
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
    real.reserve(((size_t)1u) << expression.size());
    imag.reserve(((size_t)1u) << expression.size());
    real.emplace_back(start1 ? 1 : 0, 1);
    for (unsigned char c : expression)
    {
#if 0 
      //Some unfortunate linux setups have new GCC but a std library so old that this
      //version of insert is a void function
      auto realHalf = real.insert(real.end(), imag.begin(), imag.end());
      auto imagHalf = imag.insert(imag.end(), real.begin(), realHalf);
#else
      auto realSize = real.size(), imagSize = imag.size();
      real.insert(real.end(), imag.begin(), imag.end());
      auto realHalf = real.begin() + realSize;
      imag.insert(imag.end(), real.begin(), realHalf);
      auto imagHalf = imag.begin() + imagSize;
#endif
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
    return { std::move(real), std::move(imag) };
  }

  pair<vector<Invariant>, vector<Invariant>> getInvariants(int n) {
    auto halves = possibleHalves(n);
    vector<Invariant> evens, odds;
    evens.reserve(halves.size());
    odds.reserve(halves.size());
    for (auto& v : halves) {
      auto p = multiplyOutTerm(v);
      evens.push_back(std::move(p.first));
      odds.push_back(std::move(p.second));
    }
    for (auto &invariant : evens)
      std::sort(invariant.begin(), invariant.end());
    for (auto &invariant : odds)
      std::sort(invariant.begin(), invariant.end());
    return {std::move(evens), std::move(odds)};
  }

  //returns all the sequences which have m zeros and n ones
  //Could be incorporated into calling function to save allocations
  vector<vector<unsigned char>> possibleParts(int m, int n) {
    vector<unsigned char> base(m + n, 0);
    std::fill(base.begin() + m, base.end(), (unsigned char)1);
    vector<vector<unsigned char>> out{ base };
    while (std::next_permutation(base.begin(), base.end())) {
      out.push_back(base);
    }
    return out;
  }

  //Computes the shuffle product of two invariants.
  Invariant shuffle(const Invariant& a, int alevel, const Invariant& b, int blevel) {
    int newlevel = alevel + blevel;
    vector<double> all(((size_t)1u) <<newlevel, 0.0);
    auto parts = possibleParts(alevel, blevel);
    for (const auto& aa : a) {
      for (const auto& bb : b) {
        for (const auto& pattern : parts) {
          Idx aIdx = aa.first;
          Idx bIdx = bb.first;
          size_t idx = 0;
          size_t outBit = 1;
          for (unsigned char takeB : pattern) {
            //This bit is the bottleneck. Unsure if the commented implementation is faster.
            //auto& i = (takeB != 0 ? bIdx : aIdx);
            //if (i % 2)
            //  idx += outBit;
            //i /= 2;
            if ((takeB!=0 ? bIdx : aIdx) % 2)
              idx += outBit;
            (takeB ? bIdx : aIdx) /= 2;
            outBit *= 2;
          }
          all[idx] += aa.second * bb.second;
        }
      }
    }
    Invariant out;
    for (size_t i = 0; i < all.size(); ++i) {
      if (all[i] != 0)
        out.emplace_back(i, all[i]);
    }
    return out;
  }

  enum class InvariantType {ALL, KNOWN, SVD, QR};
  //returns true on failure
  bool getWantedMethod(const std::string& input, InvariantType& t) {
    const auto npos = std::string::npos;
    bool all = (npos != input.find_first_of("aA"));
    bool known = (npos != input.find_first_of("kK"));
    bool svd = (npos != input.find_first_of("sS"));
    bool qr = (npos != input.find_first_of("qQ"));
    if (1!=(all ?1:0)+(svd?1:0)+(qr?1:0)+(known?1:0))
      return true;
    t = (all ? InvariantType::ALL : svd ? InvariantType::SVD : 
          qr ? InvariantType::QR : InvariantType::KNOWN);
    return false;
  }

/*
  struct WantedMethod {
    bool m_all = false;
    bool m_svd = false;
  };

  //returns true on failure
  bool getWantedMethod(const std::string& input, WantedMethod& wm) {
    const auto npos = std::string::npos;
    wm.m_all=(npos != input.find_first_of("aA"));
    wm.m_svd=(npos != input.find_first_of("sS"));
    return wm.m_all != wm.m_svd;
  }
*/

  //make a matrix whose columns are the given invariants
  //We know that in must either contain all odious or all evil indices, so we squish the matrix to only use
  //half the indices - we basically ignore the last bit of every index.
  void invariantsToMatrix(const vector<Invariant>& in, int level, vector<double>& out) {
    size_t d = ((size_t)1u) << level;
    d /= 2;//remove to unsquish
    out.assign(d*in.size(), 0);
    for (size_t i = 0; i < in.size(); ++i) {
      for (auto& p : in[i]) {
        size_t idx = ((size_t)p.first/2)*in.size() + i; //remove the /2 to unsquish
        out[idx] = p.second;
      }
    }
  }
  //Convert such a matrix back to invariants,
  //odiously indexed if parity is 1, evilly indexed if parity is 0
  void invariantsFromMatrix(const vector<double>& in, int level, int parity, vector<Invariant>& out) {
    size_t d = ((size_t)1u) << level;
    d /= 2;//remove to unsquish
    size_t nInvariants = in.size() / d;
    out.assign(nInvariants, {});
    for (size_t i = 0; i < d; ++i) {
      size_t ii = i;
      if (1) {//(0) to unsquish
        ii *= 2;
        //If ii's bit count has the wrong parity, we stick a 1 on the end.
        //speedup idea: use an intrinsic to count bits if available, or use a table.
        if (((size_t)parity) != std::bitset<8 * sizeof(size_t)>(ii).count() % 2)
          ++ii;
      }
      for (size_t j = 0; j < nInvariants; ++j) {
        double elt = in[i*nInvariants + j];
        if (elt != 0) //With squishing, is this always true?
          out[j].emplace_back(ii, elt);
      }
    }
  }

  //class which contains all the invariants up to a given level, and,
  //if required, all the invariants which are known as they are shuffle products 
  //of other invariants.
  class Prepared {
  public:
    //stored even then odd
    vector<vector<Invariant>> m_invariants, m_knownInvariants;
    int m_level;
    InvariantType m_type;

    Prepared(int level, InvariantType type)
    : m_level(level), m_type(type)
    {
      m_invariants.assign(level, {});
      for (int lev = 2; lev <= level; lev += 2) {
        auto p = getInvariants(lev / 2);
        m_invariants[lev - 2] = std::move(p.first);
        m_invariants[lev - 1] = std::move(p.second);
      }
      if (type != InvariantType::ALL) {
        m_knownInvariants.assign(level, {});
        //First do all the combinations of different levels
        for(int lowerLevel=2; lowerLevel<level; lowerLevel+=2)
          for (int upperLevel = lowerLevel+2; 
                upperLevel + lowerLevel <= level; upperLevel += 2) {
            for(const auto& lower: m_invariants[lowerLevel - 1])
              for (const auto& upper : m_invariants[upperLevel - 1]) {
                m_knownInvariants[upperLevel + lowerLevel - 2].push_back(
                  shuffle(lower, lowerLevel, upper, upperLevel)
                );
              }
            for (const auto& lower : m_invariants[lowerLevel - 2])
              for (const auto& upper : m_invariants[upperLevel - 2]) {
                m_knownInvariants[upperLevel + lowerLevel - 2].push_back(
                  shuffle(lower, lowerLevel, upper, upperLevel)
                );
              }
            for (const auto& lower : m_invariants[lowerLevel - 2])
              for (const auto& upper : m_invariants[upperLevel - 1]) {
                m_knownInvariants[upperLevel + lowerLevel - 1].push_back(
                  shuffle(lower, lowerLevel, upper, upperLevel)
                );
              }
            for (const auto& lower : m_invariants[lowerLevel - 1])
              for (const auto& upper : m_invariants[upperLevel - 2]) {
                m_knownInvariants[upperLevel + lowerLevel - 1].push_back(
                  shuffle(lower, lowerLevel, upper, upperLevel)
                );
              }
        }
        //Then do all the combinations of elements from the same level
        for (int lowerLevel = 2; lowerLevel + lowerLevel <= level; lowerLevel+=2) {
          const auto& source1 = m_invariants[lowerLevel - 2];
          const auto& source2 = m_invariants[lowerLevel - 1];
          for (const auto& source : { source1, source2 })
            for (size_t i = 0; i != source.size(); ++i)
              for (size_t j = i; j < source.size(); ++j)
                m_knownInvariants[(lowerLevel + lowerLevel) - 2].push_back(
                  shuffle(source[i], lowerLevel, source[j], lowerLevel)
                );
          for (const auto& lower : source1)
            for (const auto& upper : source2) {
              m_knownInvariants[lowerLevel + lowerLevel - 1].push_back(
                shuffle(lower, lowerLevel, upper, lowerLevel)
              );
            }
        }
      }
    }
  };

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
    for (auto& p : getInvariants(level / 2)) {
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

  void demoShuffle() {
    Invariant a{ { 2,1 } }; //10
    Invariant b{ { 3,1 } }; //11
    printOneTerm<2>(a);
    printOneTerm<2>(b);
    //"10 shuffle 11" is 1011+(2)1101+(3)1110
    Invariant out = shuffle(a, 2, b, 2);
    printOneTerm<4>(out);
  }
#endif
}

#endif
