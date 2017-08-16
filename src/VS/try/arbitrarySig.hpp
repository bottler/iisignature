#pragma once

#include "bch.hpp"
#include "logsig.hpp"

//printArbitrarySig prints out the elements of a signature
//in terms of the basis elements of its log signature
//I use it to generate the data on a big beamer slide.
//This code is not designed to be efficient - lots of copying.
namespace ArbitrarySig {
  using namespace std;

  void printCoefficientAsLetter(const Coefficient& d, std::ostream& o) {
    if (d.m_details.empty()) {
      o << "0 ";
      return;
    }
    for (size_t ic = 0; ic<d.m_details.size(); ++ic) {
      const auto& c = d.m_details[ic];
      if (c.second >= 0)
        o << "+";
      if (c.second == -1 && !c.first.empty())
        o << "-";
      if (c.second != 1 && c.second != -1 || (c.first.empty()))
        o << c.second;
      for (const auto& i : c.first) {
        if (i.m_index > 25)
          o << "?";
        else
          o << static_cast<char>(i.m_index + 'a');
      }
    }
    o << " ";
  }

  class ArbitrarySignature {
  public:
    vector<vector<Coefficient>> m_data;
    ArbitrarySignature(int d, int m) {
      m_data.resize(m);
      m_data[0].resize(d);
      for (int i = 1; i < m; ++i)
        m_data[i].resize(m_data[i - 1].size()*d);
    }
    void print(ostream& o) const {
      for (auto& l : m_data) {
        for (auto& p : l)
          printCoefficientAsLetter(p, o);
        o << "\n";
      }
    }
  };

  ArbitrarySignature concatenateWith_zeroFirstLevel(int d, int m,
    const ArbitrarySignature& a,
    const ArbitrarySignature& b) {
    ArbitrarySignature out(d, m);
    for (int level = m; level>0; --level) {
      for (int alevel = level - 1; alevel>0; --alevel) {
        int blevel = level - alevel;
        auto& aa = a.m_data[alevel - 1];
        auto& bb = b.m_data[blevel - 1];
        auto dest = out.m_data[level - 1].begin();
        for (const Coefficient& c : aa) {
          for (const Coefficient& cc : bb) {
            sumCoefficients(*(dest++), productCoefficients(cc, c));
          }
        }
      }
    }
    return out;
  }

  //this is just like exponentiateTensor in test_sig.py
  ArbitrarySignature exponentiate(int d, int m, const ArbitrarySignature& a) {
    ArbitrarySignature out = a;
    vector<ArbitrarySignature> products{ out };
    for (int mm = 2; mm <= m; ++mm) {
      auto t = concatenateWith_zeroFirstLevel(d, m, a, products.back());
      for (auto& j : t.m_data)
        for (auto& co : j)
          for (auto& y : co.m_details)
            y.second *= 1.0 / mm;
      for (int i = 0; i < m; ++i) {
        for (size_t j = 0; j < out.m_data[i].size(); ++j) {
          auto tempCopy = t.m_data[i][j];
          sumCoefficients(out.m_data[i][j], std::move(tempCopy));
        }
      }
      products.push_back(t);
    }
    return out;
  }

  void printArbitrarySig(const int d, const int m) {
    using namespace IISignature_algebra;
    BasisPool basisPool(LieBasis::Lyndon);
    vector<BasisElt*> elts;
    auto list = makeListOfBasisElts(basisPool, d, m);
    auto logsig = std::make_unique<Polynomial>();
    int idx = 0;
    for (auto& v : list) {
      logsig->m_data.push_back({});
      for (auto elt : v) {
        //char c = 'a' + idx;
        logsig->m_data.back().push_back(make_pair(elt, basicCoeff(idx)));
        ++idx;
        elts.push_back(elt);
      }
    }
    printPolynomial(*logsig, std::cout);
    vector<size_t> sigLevelSizes{ (size_t)d };
    for (int level = 2; level <= m; ++level)
      sigLevelSizes.push_back(d*sigLevelSizes.back());
    auto mappingMatrix = makeMappingMatrix(d, m, basisPool, elts, sigLevelSizes);
    ArbitrarySignature tensorspace(d, m);
    for (int level = 1; level <= m; ++level) {
      auto& logdata = logsig->m_data[level - 1];
      auto& mapping = mappingMatrix[level - 1];
      auto& outlevel = tensorspace.m_data[level - 1];
      for (auto& p : logdata) {
        const auto& sparseMatrix = lookupInFlatMap(mapping, p.first);
        for (auto& pos : sparseMatrix) {
          auto& dest = outlevel[pos.first];
          if (dest.m_details.empty()) {
            dest = p.second;
            for (auto& x : dest.m_details)
              x.second *= pos.second;
          }
          else {
            auto tempcopy = p.second;
            for (auto& x : tempcopy.m_details)
              x.second *= pos.second;
            sumCoefficients(dest, std::move(tempcopy));
          }
        }
      }
    }
    tensorspace.print(std::cout);
    auto sig = exponentiate(d, m, tensorspace);
    //we multiply each level by the factorial so that all numbers come out as an integer.
    int prod = 1;
    for (int level = 2; level <= m; ++level) {
      //multiply by level!
      prod *= level;
      for (auto& co : sig.m_data[level - 1])
        for (auto& y : co.m_details)
          y.second *= prod;

    }
    sig.print(std::cout);
  }
}