#include<iostream>
#include<fstream>
#include<thread>
#include<iomanip>
#include<chrono>
#include<set>
#include"logsig.hpp"
#include "arbitrarySig.hpp"
#include "rotationalInvariants.hpp"
#include "utils.h"
using namespace std;

void interrupt() {}
void interrupt1() {
  while (1) {
    ifstream ifs("C:\\play\\foo1.txt");
    if (!ifs.is_open())
      return;
    ifs.close();
    //exit(0);
    std::cout << "interrupted" << std::endl;
    using namespace std::chrono_literals;
    this_thread::sleep_for(1s);
  }
}

void setupGlobal() {
  ifstream f("c:\\play\\iisignature\\iisignature_data\\bchLyndon20.dat");
  auto s = new std::string(istreambuf_iterator<char>(f), {});
  g_bchLyndon20_dat = s->c_str();
}

class SecondsCounter
{
  const std::chrono::steady_clock::time_point start;
  double& output;
public:
  SecondsCounter(double& output) :start(std::chrono::steady_clock::now()), output(output) {}
  ~SecondsCounter() { output = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() / 1000.0; }
};

void trial() {
  setupGlobal();
  LogSigFunction lsf(LieBasis::Lyndon);
  WantedMethods wm;
  wm.m_expanded = wm.m_compiled_bch = wm.m_log_of_signature = wm.m_simple_bch = false;
  wm.m_compiled_bch = true;
  makeLogSigFunction(2, 2, lsf, wm, interrupt);
  vector<double> signature{ 3,3,3.0 };
  vector<double> displacement{ 1,17.0 };
  lsf.m_f->go(signature.data(), displacement.data());
}

void countFlopsInCompiledCode() {
  setupGlobal();
  WantedMethods wm;
  wm.m_expanded = wm.m_compiled_bch = wm.m_log_of_signature = wm.m_simple_bch = false;
  wm.m_compiled_bch = true;
  for (int d : {2}) {
    for (int m : {2, 3, 4, 5, 6, 7, 8, 9}) {
      LogSigFunction lsf(LieBasis::Lyndon);
      makeLogSigFunction(d, m, lsf, wm, interrupt);
      auto flops = lsf.m_f->m_stats.m_flops;
      std::cout << d << " " << m << " " << flops << std::endl;
    }
  }
}

void trySVD() {
  setupGlobal();
  LogSigFunction lsf(LieBasis::Lyndon);
  WantedMethods wm;
  wm.m_expanded =wm.m_compiled_bch= wm.m_log_of_signature = wm.m_simple_bch = false;
  wm.m_log_of_signature = true;
  //makeLogSigFunction(103, 4, lsf, wm, interrupt1);
  makeLogSigFunction(2, 5, lsf, wm, interrupt);
}

void compareHL() {
  setupGlobal();
  WantedMethods wm;
  wm.m_expanded = wm.m_compiled_bch = wm.m_log_of_signature = wm.m_simple_bch = false;
  wm.m_compiled_bch = true;
  for (int i = 0; i < 1; ++i) {
    LogSigFunction lyndon(LieBasis::Lyndon);
    LogSigFunction hall(LieBasis::StandardHall);
    int d = 2;
    int m = 7;
    makeLogSigFunction(d, m, lyndon, wm, interrupt);
    makeLogSigFunction(d, m, hall, wm, interrupt);
    std::cout << "Lyndon: " << lyndon.m_f->m_m.used() << "\n";
    std::cout << "Hall: " << hall.m_f->m_m.used() << "\n";
  }
}

//Look at timing difference using different basis
double doTrianglesHelp() {
  setupGlobal();
  WantedMethods wm;
  wm.m_expanded = wm.m_compiled_bch = wm.m_log_of_signature = wm.m_simple_bch = false;
  wm.m_log_of_signature = true;
  LogSigFunction lsf(LieBasis::Lyndon);
  //LogSigFunction lsf(LieBasis::StandardHall);
  int d = 2;
  int m = 5;
  makeLogSigFunction(d, m, lsf, wm, interrupt);
  vector<double> seg(d);
  vector<double> output((size_t)std::pow(d, m + 1));
  double out = 0;
  
  {
    SecondsCounter sc(out);
    for (int i = 0; i < 100; ++i) {
      CalcSignature::Signature sig;
      sig.sigOfSegment(d, m, &output[0]);
      logTensorHorner(sig);
      projectExpandedLogSigToBasis(&output[0], &lsf, sig);
      out+=output[0] ;//just to confound the optimizer
    }
  }
  std::cout << "\ntime taken (s): " << out << "\n";
  return out;
}

void __fastcall foo(double* a, const double* b, double* c);
typedef void(__fastcall *F)(double*, const double*, double*);

int main1() { 
  trySVD();
  //trial();
  double a = 3, b = 4, v = 4;
  foo(&a, &b, &v);
  F ff = foo;
  //cout << 3 << "\n";
  return 0;
}

//returns all the max^length members of the set {0 .. max-1}^length
//in lexicographic order
//as sequences in one long vector. The output has size length*max^length.
std::vector<int> getAllSequences(int max, int length) {
  std::vector<int> all(length * (int)std::pow(max, length));
  vector<int> buffer(length,0);
  auto outit = all.begin();
  while (1) {
    for (int i : buffer)
      *(outit++) = i;
    bool found = false;
    for (int offsetToChange = length - 1; offsetToChange >= 0; --offsetToChange) {
      if (buffer[offsetToChange] < max-1) {
        ++buffer[offsetToChange];
        for (int i = offsetToChange + 1; i < length; ++i)
          buffer[i] = 0;
        found = true;
        break;
      }
    }
    if (!found)
      return all;
  }
}

bool isSymmetric(const vector<vector<double>>& squareMatrix) {
  size_t n = squareMatrix.size();
  double maxdiff = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) {
      double diff = std::fabs(squareMatrix[j][i] - squareMatrix[i][j]);
      maxdiff = std::max(diff, maxdiff);
    }
  }
  std::cout << "biggest assymetry: "<< maxdiff << "\n";
  return false;
}

//if you have no indeterminates, c will be a number, so get it
double numberFromCoeff(const Coefficient& coeff) {
  if (coeff.m_details.empty())
    return 0;
  if (coeff.m_details.size() == 1 && coeff.m_details[0].first.empty())
    return coeff.m_details[0].second;
  throw std::runtime_error("a coeff which isn't just a number shouldn't be here");
}

//This function prints out the matrix from the mth level of tensor space to itself
//corresponding to the dynkin map.
//It is abundantly clear that this matrix is not symmetric
//and therefore the dynkin map is not an *orthogonal* projection (in the obvious basis).
void dynkinExperiment(const int d, const int m, bool p1, bool p2) {
  using namespace IISignature_algebra;
  BasisPool s(LieBasis::Lyndon);
  auto list = makeListOfBasisElts(s, d, m);
  vector<BasisElt*> elts;
  for (auto& v : list)
    for (auto elt : v)
      elts.push_back(elt);
  vector<size_t> sigLevelSizes{ (size_t)d };
  for (int level = 2; level <= m; ++level)
    sigLevelSizes.push_back(d*sigLevelSizes.back());
  auto mappingMatrix = makeMappingMatrix(d, m, s, elts, sigLevelSizes);
  const auto& map = mappingMatrix[m - 1];
  vector<vector<double>> fullMatrix(sigLevelSizes.back());
  for (auto& v : fullMatrix)
    v.resize(sigLevelSizes.back());
  auto& letters = list[0];
  vector<int> allSeqs = getAllSequences(d, m);
  int nWords = ((int)allSeqs.size()) / m;
  for (int i = 0; i < nWords; ++i)
  {
    int offset = i*m;
    BasisElt* firstLetter = letters[allSeqs[offset]];
    auto poly = polynomialOfBasisElt(firstLetter);
    auto poly2 = polynomialOfBasisElt(letters[allSeqs[offset + m - 1]]);
    for (int j = 1; j < m; ++j) {
      auto n = productPolynomials(s, &*poly, &*polynomialOfBasisElt(letters[allSeqs[offset + j]]), 1 + j);
      poly = move(n);
      auto n2 = productPolynomials(s, &*polynomialOfBasisElt(letters[allSeqs[offset + m-1-j]]), &*poly2, 1 + j);
      poly2 = move(n2);
    }
    if (!p1)
      poly.reset();
    if (!p2)
      poly2.reset();
    if (poly)//poly)
      for (auto &l : poly->m_data)
        for (auto& ll : l) {
          double co = numberFromCoeff(ll.second);
          for (auto p : lookupInFlatMap(map,ll.first))
            fullMatrix[i][p.first] += co*p.second;
        }
    if (poly2)//poly2)
      for (auto &l : poly2->m_data)
        for (auto& ll : l) {
          double co = numberFromCoeff(ll.second);
          for (auto p : lookupInFlatMap(map, ll.first))
            fullMatrix[i][p.first] += co*p.second;
        }
  }
  int pos = 0;
  if(1)
  for (auto &v : fullMatrix) {
    for (auto dd : v)
      std::cout << std::setw(2)<< dd << " ";
    if (1)
      for (int i = 0; i < m; ++i)
        std::cout << allSeqs[pos++];
    std::cout << "\n";
  }
  isSymmetric(fullMatrix);
}


size_t countAnagrams(const vector<Letter>& sortedLetters) {
  size_t numerator = 1;
  size_t currentDenominator = 1;
  size_t denominator = 1;
  size_t thisNumber = 1;
  const size_t size = sortedLetters.size();
  for (size_t i = 0; i < size; ++i) {
    if (i == 0 || sortedLetters[i] != sortedLetters[i - 1]) {
      denominator *= currentDenominator;
      currentDenominator = thisNumber = 1;
    }
    else {
      ++thisNumber;
      currentDenominator *= thisNumber;
    }
    numerator *= i + 1;
  }
  denominator *= currentDenominator;
  if (numerator%denominator != 0)
    throw std::runtime_error("math error");
  return numerator / denominator;
}

void anagramSetCountings() {
  using namespace IISignature_algebra;
  for (int d : {2, 3, 4, 5, 6, 7})
    for (int m : {2, 3, 4, 5, 6, 7, 8}) {
      std::cout << "d=" << d << " m=" << m;
      BasisPool s(LieBasis::Lyndon);
      auto list = makeListOfBasisElts(s, d, m);
      vector<BasisElt*> elts;
      for (auto& v : list)
        for (auto elt : v)
          elts.push_back(elt);
      vector<size_t> sigLevelSizes{ (size_t)d };
      for (int level = 2; level <= m; ++level)
        sigLevelSizes.push_back(d*sigLevelSizes.back());
      auto mappingMatrix = makeMappingMatrix(d, m, s, elts, sigLevelSizes);
      BasisEltToIndex basisEltToIndex;
      LetterOrderToBE letterOrderToBE;
      analyseMappingMatrixLevel(mappingMatrix, m, letterOrderToBE, basisEltToIndex);
      size_t total = 0;
      for (auto& i : letterOrderToBE) {
        size_t nLyndonWords = i.second.size();
        size_t nWords = countAnagrams(i.first);
        //std::cout << "\n" << countAnagrams(i.first) << ", " << nLyndonWords;
        total += nLyndonWords * nWords;
      }
      std::cout << ": " << total << "\n";
      //std::cout << "\n";
    }
}

std::vector<Letter> indexToWord(size_t index, int d, int m) {
  std::vector<Letter> o;
  for (int i = 0; i < m; ++i) {
    Letter dig = (Letter)(index % d);
    index = index / d;
    o.push_back(dig);
  }
  std::reverse(o.begin(), o.end());
  return o;
}

bool equalLetters(const BasisElt* elt, const vector<int>& counts) {
  vector<int> mycounts(counts.size());
  elt->iterateOverLetters([&](Letter l){
    ++mycounts[l];
  });
  return mycounts == counts;
}

size_t countLyndonWords(const vector<int>& counts){
    vector<Letter> myletters;
  for (Letter i = 0; i < counts.size(); ++i)
    for (int j = 0; j < counts[i]; ++j)
      myletters.push_back(i);
  int m = (int)myletters.size();
  int d = 1 + *std::max_element(myletters.begin(), myletters.end());
  using namespace IISignature_algebra;
  //BasisPool s(LieBasis::Lyndon);
  BasisPool s(LieBasis::StandardHall);
  auto list = makeListOfBasisElts(s, d, m);
  size_t out = 0;
  for (auto elt : list.back()) {
    if (equalLetters(elt, counts))
      ++out;
  }
  return out;
}

size_t countLyndonWordsWithTwos(size_t nTwos) {
  vector<int> counts(2, (int)nTwos);
  return countLyndonWords(counts);
}

void printAMappingMatrix() {
  //vector<Letter> myletters{ 0,0,1,1,2,2 };
  //vector<Letter> myletters{ 0,0,1,1,2,2,2 };
  //vector<Letter> myletters{ 0,0,0,0,1,1,1,1 };
  vector<Letter> myletters{ 0,0,0,1,1,1,2,2,2,2 };
  //vector<Letter> myletters{ 0,0,1,2 };
  //vector<Letter> myletters{ 0,1,2};
  if (!std::is_sorted(myletters.begin(), myletters.end()) || myletters.at(0) != 0)
    throw "myletters must be sorted";
  int m = (int)myletters.size();
  int d = 1 + *std::max_element(myletters.begin(), myletters.end());
  using namespace IISignature_algebra;
  //BasisPool s(LieBasis::Lyndon);
  BasisPool s(LieBasis::StandardHall);
  bool printBrackets = 1;// LieBasis::StandardHall == s.m_basis;
  auto list = makeListOfBasisElts(s, d, m);
  vector<BasisElt*> elts;
  for (auto& v : list)
    for (auto elt : v)
      elts.push_back(elt);
  vector<size_t> sigLevelSizes{ (size_t)d };
  for (int level = 2; level <= m; ++level)
    sigLevelSizes.push_back(d*sigLevelSizes.back());
  auto mappingMatrix = makeMappingMatrix(d, m, s, elts, sigLevelSizes);
  BasisEltToIndex basisEltToIndex;
  LetterOrderToBE letterOrderToBE;
  analyseMappingMatrixLevel(mappingMatrix, m, letterOrderToBE, basisEltToIndex);
  auto v = lookupInFlatMap(letterOrderToBE, myletters);
  std::set<size_t> usedTensorIndicesS;
  for (auto elt : v) {
    for (auto& i : lookupInFlatMap(mappingMatrix.back(),elt))
      usedTensorIndicesS.insert(i.first);
    //printBasisEltDigits(*elt, std::cout);
    //std::cout << "\n";
  }
  //vector<size_t> usedTensorIndices(usedTensorIndicesS.begin(), usedTensorIndicesS.end());
  std::map<size_t, size_t> bigIdx2SmallIdx;
  {
    size_t idx = 0;
    for (auto i : usedTensorIndicesS)
      bigIdx2SmallIdx[i] = idx++;
  }
  std::ofstream output("foo.txt");
  vector<vector<Letter>> tensorLetters;
  for (auto i : usedTensorIndicesS)
    tensorLetters.push_back(indexToWord(i, d, m));
  size_t initialSpaces = 1 + m;
  if (printBrackets) {
    initialSpaces += 3 * (m - 1);//two brackets and a comma
  }
  for (int i = 0; i < m; ++i) {
    for (size_t j = 0; j < initialSpaces; ++j)
      output << " ";
    for (const auto& j : tensorLetters)
      output << (char)('1' + j[i]);
    output << "\n";
  }
  for (auto elt : v) {
    vector<float> out(usedTensorIndicesS.size());
    auto& expanded = lookupInFlatMap(mappingMatrix.back(), elt);
    for (auto& i : expanded)
      out[bigIdx2SmallIdx.at(i.first)] = i.second;
    if(!printBrackets)
      printBasisEltDigits(*elt, output);
    else
      printBasisEltBracketsDigits(*elt, output);
    output << " ";
    vector<Letter> elt_letters;
    elt->iterateOverLetters([&](Letter l) {elt_letters.push_back(l); });
    //std::reverse(elt_letters.begin(), elt_letters.end());
    size_t elt_idx = 0;
    for (Letter l : elt_letters) {
      elt_idx *= d;
      elt_idx += l;
    }
    int elt_small_idx = -1;
    if (s.m_basis==LieBasis::Lyndon)
      elt_small_idx = (int)bigIdx2SmallIdx.at(elt_idx);
      
    for (size_t j = 0; j != out.size(); ++j) {
      float i = out[j];
      float nice = i < 0 ? -i : i;
      nice = (nice <= 9) ? nice : 9;
      //output << " ";
      if (s.m_basis == LieBasis::Lyndon && elt_small_idx == (int)j) {
        std::cout << i << "\n";
        output << (nice != 1 ? "*" : "#");
      }
      else
        output << nice;
    }
    output << "\n";
  }
}

void timeRotInv() {
  double out=0;
  double sum = 0;
  {
    SecondsCounter s(out);
    int level = 16;
    for (int lev = 2; lev <= level; lev += 2) {
      auto invs = RotationalInvariants::getInvariants(lev / 2);
      sum += invs.first[1][0].first;
    }
  }
  std::cout << "time taken (s): " << out << "\n";
  std::cout << sum << "\n";//just to confound the optimizer
}


void timeRotInv2() {
  double out = 0;
  double sum = 0;
  {
    SecondsCounter s(out);
    int level = 14;
    RotationalInvariants::Prepared(level, RotationalInvariants::InvariantType::KNOWN);
  }
  std::cout << "time taken (s): " << out << "\n";
  std::cout << sum << "\n";//just to confound the optimizer
}

//Check that the Mem object cleans up properly.
void memLeakChecker() {
  for (int i = 0; i < 2000000; ++i) {
    Mem m(20*1024*1024);
    m.push(2);
    if (i % 100 == 0)
      std::cout << "a\n";
  }
  int ii;
  std::cin >> ii;
  std::cout << "ok\n";
}

void tryLogBackwards() {
  int d = 3, m = 6;
  CalcSignature::Signature sig, der;
  vector<double> segment(d);
  segment[0] = 3;
  segment[1] = 2;
  //sig.sigOfNothing(d, m);
  sig.sigOfSegment(d, m, segment.data());
  if (false) {
    der.sigOfNothing(d, m);
    for (int i = 0; i < m; ++i)
      std::fill(der.m_data[i].begin(), der.m_data[i].end(), (CalcSignature::Number)1.0);
    //std::fill(sig.m_data[0].begin(), sig.m_data[0].end(), (CalcSignature::Number)3.0);
  }
  else
  {
    WantedMethods wm;
    wm.m_expanded = wm.m_compiled_bch = wm.m_log_of_signature = wm.m_simple_bch = false;
    wm.m_log_of_signature = true;
    LogSigFunction lsf(LieBasis::Lyndon);
    makeLogSigFunction(d, m, lsf, wm, interrupt);
    vector<double> logsig(lsf.m_basisElements.size());
    projectExpandedLogSigToBasisBackwards(logsig.data(), &lsf, der);
  }
  //sig.m_data[0].at(1) = 32;
  logBackwards(der, sig);
  for (auto x : der.m_data[0])
    std::cout << x << "\n";
}

int main() {
  restrictWorkingSet(2000);
  //tryLogBackwards();
  countFlopsInCompiledCode();
  //anagramSetCountings();
  //std::cout << countLyndonWords({ 3,3,4 }) << "\n";
  //printAMappingMatrix();
  //compareHL();
  //memLeakChecker();
  //RotationalInvariants::printAsMatrix();
  //timeRotInv2();
  //RotationalInvariants::demoShuffle();
  //trial();
  //printListOfLyndonWords(2, 5);
  //ArbitrarySig::printArbitrarySig(3, 6);
  //dynkinExperiment(2, 4, 1, 0);
  //dynkinExperiment(2, 4, 0, 1);
  //dynkinExperiment(2, 4, 1, 1);
  //trySVD();
  //doTrianglesHelp();
  //setupGlobal();
  //calcFla(2, 4, interrupt);
  return 0;
}
