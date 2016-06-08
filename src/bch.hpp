#ifndef BCH_H
#define BCH_H
#include<algorithm>
#include<functional>
#include<iostream>
#include<iterator>
#include<memory>
#include<utility>
#include<vector>

#include "readBCHCoeffs.hpp"

//amalgamate_adjacent(v.begin(),v.end(),tojoin,amalgamater)
//iterates over v and when finding a range [i..j) for which tojoin(*i,*n) for all n in (i,j), replaces it with 
//  (the single element *i if amalgamater(i,j) and nothing otherwise)
//returns one past the last - removes everything made unnecessary

//I.e.: If you have a sequence where you want to merge any sequence of adjacent elements which have something in common so that they become
//either a single element or nothing, use this function where tojoin identifies elements with the "something in common" and amalgamater makes
//what they become and returns whether they become anything at all.

//Could use swaps instead of assignment and potentially save.

template<typename I, typename F1, typename F2>
  I amalgamate_adjacent(I a, I b, F1&& tojoin, F2&& amalgamater){
  I dest = a;
  while(a!=b){
    I temp = a;
    int rangelength = 0;
    while(++temp!=b && tojoin(*a,*temp)){
      ++rangelength;
    }
    if(rangelength){
      if(amalgamater(a,temp))
	*dest++ = *a;
    }
    else if(dest!=a)
      *dest++ = *a;
    else
      ++dest;
    a=temp;
  }
  return dest;
}

//amalgamate_adjacent_pairs(v.begin(),v.end(),tojoin,amalgamater)
//iterates over v and  if tojoin(*i,*(i+1)), replaces *i and *(i+1) with the single element *i if amalgamater(*i,*(i+1)) and nothing otherwise
//returns one past the last - removes everything made unnecessary

//This is the same as the previous function for the case where we know already that there won't be sequences longer 
//than 2 elements with the something in common,
//so we can simplify a bit and we can pass 2 elements instead of a pair of iterators into amalgamater.

template<typename I, typename F1, typename F2>
I amalgamate_adjacent_pairs(I a, I b, F1&& tojoin, F2&& amalgamater){
  I dest = a;
  while(a!=b){
    I temp = a;
    int rangelength = 0;
    if(++temp!=b && tojoin(*a,*temp)){
      ++rangelength;
    }
    if(rangelength){
      if(amalgamater(*a,*temp))
	*dest++ = *a;
      ++temp;
    }
    else if(dest!=a)
      *dest++ = *a;
    else
      ++dest;
    a=temp;
  }
  return dest;
}

//need transform version

const int maxDim = 255; //so I can use unsigned char for letter
using Letter = unsigned char;

class LyndonWord{
  friend class LyndonWordIterator;
 public:
  LyndonWord(Letter l): m_left(0), m_letter(l){}
  LyndonWord(const LyndonWord& left, const LyndonWord& right)
    :m_left(&left), m_right(&right){}
  bool isLetter() const {return !m_left;}
  Letter getLetter() const {return m_letter;}
  const LyndonWord* getLeft() const {return m_left;}
  const LyndonWord* getRight() const {return m_right;}
  template<typename T>
  void iterateOverLetters(T&& f) const {
    if(isLetter())
      f(getLetter());
    else{
      getLeft()->iterateOverLetters(f);
      getRight()->iterateOverLetters(f);
    }    
  }
  int length() const {
    int o = 0;
    iterateOverLetters([&](Letter){++o;});
    return o;
  }
  bool isEqual(const LyndonWord& o) const {
    if(isLetter() != o.isLetter())
      return false;
    if(isLetter())
      return getLetter()==o.getLetter();
    return getLeft()->isEqual(*o.getLeft()) && getRight()->isEqual(*o.getRight());
  }
 private:
  const LyndonWord* m_left; //nullptr if we are a letter
  union{
    const LyndonWord* m_right;
    Letter m_letter;
  };
};

//iterate over letters from left to right
//Beware that if you copy an object of this type (except an end), only one of the copies is usable,
//because they share the same working space.
class LyndonWordIterator : public std::iterator<std::forward_iterator_tag, Letter>{
 public:
  Letter operator*() const {return m_thisLetter->getLetter();}
  LyndonWordIterator(const LyndonWord* word, std::vector<const LyndonWord*>& tempSpace){
    tempSpace.clear();
    m_vec=&tempSpace;
    while(!word->isLetter()){
      m_vec->push_back(word);
      word=word->m_left;
    }
    m_thisLetter=word;
  }
  LyndonWordIterator() : m_thisLetter(nullptr), m_vec(nullptr){}
  bool operator== (const LyndonWordIterator& o)const {return m_thisLetter == o.m_thisLetter;}
  bool operator!= (const LyndonWordIterator& o)const {return m_thisLetter != o.m_thisLetter;}
  bool isEnd()const {return !m_thisLetter;}
  void operator++(){
    while(!m_vec->empty() && m_thisLetter==m_vec->back()->m_right){
      m_thisLetter = m_vec->back();
      m_vec->pop_back();
    }
    //now I want the left part of the rightmost child of m_vec.back()
    if(m_vec->empty())
      m_thisLetter=nullptr;
    else{
      m_thisLetter= m_vec->back()->m_right;
      while(!m_thisLetter->isLetter()){
	m_vec->push_back(m_thisLetter);
	m_thisLetter=m_thisLetter->m_left;
      }
    }
  }
 private:
  const LyndonWord* m_thisLetter;
  std::vector<const LyndonWord*>* m_vec;
};

void printLetterAsDigit(Letter l, std::ostream& o){
  if(l>8)
    o<<'?';
  else
    o<<(char)('1'+l);
}

void printLyndonWordDigits(const LyndonWord& w, std::ostream& o){
  /*
  LyndonWordIterator i(&w), end;
  while(!(i==end)){
    printLetterAsDigit(*i,o);
    ++i;
  }
  */
  w.iterateOverLetters([&](Letter l){printLetterAsDigit(l,o);});
}
void printLyndonWordBracketsDigits(const LyndonWord& w, std::ostream& o){
  if(w.isLetter())
    printLetterAsDigit(w.getLetter(),o);
  else{
    o<<"[";
    printLyndonWordBracketsDigits(*w.getLeft(),o);
    o<<",";
    printLyndonWordBracketsDigits(*w.getRight(),o);
    o<<"]";
  }
}

//basically a pool for Lyndon words
class WordPool{
  void* getSpace(){
    if(m_used + objectSize>=eachLength){
      m_used=0;
      m_space.push_back({});
      m_space.back().resize(eachLength);
    }
    void** out = m_space.back().data()+m_used;
    m_used+=objectSize;
    return (void*) out;
  }
  std::vector<std::vector<void*>> m_space;
  std::vector<const LyndonWord*> m_spaceForIterator1, m_spaceForIterator2;
 public:
  static const int eachLength = 2000;
  static const int objectSize = (sizeof(LyndonWord)+sizeof(void*)-1)/sizeof(void*);
  int m_used = eachLength+2;//so we know we need to allocate at the start
  LyndonWord* newLyndonWordFromLetter(Letter l){
    static_assert(2==objectSize, "bad objectSize");
    static_assert(eachLength%objectSize==0, "bad eachLength");
    return new(getSpace()) LyndonWord(l);
  }
  LyndonWord* newLyndonWord(const LyndonWord& left, const LyndonWord& right){
    return new(getSpace()) LyndonWord(left,right);
  }
  bool lexicographicLess(const LyndonWord* l, const LyndonWord* r){
    LyndonWordIterator lit(l,m_spaceForIterator1), rit(r,m_spaceForIterator2), end;
    //  return std::lexicographical_compare(std::ref(lit),std::ref(end),std::ref(rit),std::ref(end));
    return std::lexicographical_compare(lit,end,rit,end);
  }
};

std::vector<LyndonWord*> makeListOfLyndonWords(WordPool& s, int d,int m){
  std::vector<std::vector<LyndonWord*>> words(m);
  words[0].resize(d);
  for(Letter i=0; i<d; ++i){
    words[0][i] = s.newLyndonWordFromLetter(i);
  }
  for(int level=2; level<=m; ++level){
    for(int leftlength = 1; leftlength<level; ++leftlength){
      const int rightlength = level-leftlength;
      for(LyndonWord* left : words[leftlength-1])
	for(LyndonWord* right : words[rightlength-1])
	  if(s.lexicographicLess(left,right) && (left->isLetter() || left->getRight()==right || 
					       !s.lexicographicLess(left->getRight(),right)))
	    words[level-1].push_back(s.newLyndonWord(*left,*right));
    }
  }
  //return words;
  std::vector<LyndonWord*> out;
  for(const auto& v : words){
    out.insert(out.end(),v.begin(),v.end());
  }
  return out;
}

class Input{
public:
  int m_index;
  bool operator==(const Input& o)const {return m_index==o.m_index;}
  bool operator<(const Input& o)const {return m_index<o.m_index;}
};

//worry later about pooling these allocations

class Coefficient{
public:
  //the vector<Input> is sorted
  std::vector<std::pair<std::vector<Input>,double>> m_details; //This represents a sum of a product of a load of inputs and a constant. 
};

//we could have an in-place version too
Coefficient productCoefficients (const Coefficient& a, const Coefficient& b){
  Coefficient o;
  auto& out = o.m_details;
  out.reserve(a.m_details.size()*b.m_details.size());
  for(const auto& x : a.m_details)
    for(const auto& y : b.m_details){
      out.push_back({});
      std::merge(x.first.begin(),x.first.end(),y.first.begin(),y.first.end(),std::back_inserter(out.back().first));
      out.back().second = x.second * y.second;
    }
  std::sort(out.begin(), out.end());
  using A = std::pair<std::vector<Input>,double>;
  using I = std::vector<A>::iterator;
  out.erase(amalgamate_adjacent(out.begin(),out.end(),[](const A& a, const A& b){return a.first==b.first;},
				    [](I a, I b){
				      double total = 0;
				      std::for_each(a,b,[&](const A& a){total += a.second;});
				      a->second = total;
				      return std::fabs(total)>0.00000001;})
	  ,out.end());
  //Todo: we could have a member Coefficient::m_size which was the important bit of m_details, and not erase vectors whose memory 
  //we might want to reuse later. Same with Polynomial
  return o;
}

void sumCoefficients(Coefficient& lhs, Coefficient& rhs){//make the lhs be lhs+rhs, no care about rhs
  auto& a = lhs.m_details;
  auto& b = rhs.m_details;

  size_t ss = a.size();
  a.reserve(a.size()+b.size());
  std::move(b.begin(),b.end(),std::back_inserter(a));
  std::inplace_merge(a.begin(),a.begin()+ss,a.end());

  using A = std::pair<std::vector<Input>,double>;
  a.erase(amalgamate_adjacent_pairs(a.begin(), a.end(),[](const A& a, const A& b){return a.first==b.first;},
				    [](A& a, const A& b){
				      double total = a.second + b.second;
				      a.second = total;
				      return std::fabs(total)>0.00000001;
				    })
	  ,a.end());
}

class Polynomial{
public:
  std::vector<std::pair<const LyndonWord*,Coefficient>> m_data; //kept lexicographic
};

std::ostream& printPolynomial(Polynomial& poly, std::ostream& o){
  for(auto &m : poly.m_data){
    printLyndonWordDigits(*m.first,o);
    o<<" ";
    for(const auto& c : m.second.m_details){
      for(const auto& i : c.first){
	o<<"["<<i.m_index<<"]";
      }
      o<<c.second<<"\n";
    }
  }
  return o;
}

std::unique_ptr<Polynomial> polynomialOfWord(const LyndonWord* w){
    auto a = std::unique_ptr<Polynomial>(new Polynomial);
    a->m_data.resize(1);
    a->m_data[0].first = w;
    a->m_data[0].second.m_details.resize(1);
    a->m_data[0].second.m_details[0].second=1;
    return a;
}

using Term = std::pair<const LyndonWord*,Coefficient>;

struct TermLess{
  WordPool& m_s;
  TermLess(WordPool& s):m_s(s){}
  bool operator() (const Term& a, const Term& b) const {
    return m_s.lexicographicLess(a.first, b.first);
  }
};
void sumPolynomials(WordPool& s, Polynomial& lhs, Polynomial& rhs){//make the lhs be lhs+rhs, rhs is preserved
  auto& a = lhs.m_data;
  auto& b = rhs.m_data;

  size_t ss = a.size();
  //if b has enough capacity, maybe we should detect and swap them here. similar in sumCoefficients.
  a.reserve(a.size()+b.size());
  std::move(b.begin(),b.end(),std::back_inserter(a));
  std::inplace_merge(a.begin(),a.begin()+ss,a.end(),TermLess(s));

  a.erase(amalgamate_adjacent_pairs(a.begin(), a.end(),[](const Term& a, const Term& b){return a.first->isEqual(*b.first);},
				    [](Term& a, Term& b){ 
				      sumCoefficients(a.second,b.second); 
				      return !a.second.m_details.empty();})
	  ,a.end());
}

std::unique_ptr<Polynomial> productLyndonWords(WordPool& s, const LyndonWord& a, const LyndonWord& b, int maxLength, bool check=true);

//returns 0 or a new Polynomial, taking ownership
//std::unique_ptr<Polynomial> productPolynomials(WordPool& s, std::unique_ptr<Polynomial> x, std::unique_ptr<Polynomial> y, int maxLength){
std::unique_ptr<Polynomial> productPolynomials(WordPool& s, const Polynomial* x, const Polynomial* y, int maxLength){
  if(!x || !y){
    return nullptr;
  }
  auto out = std::unique_ptr<Polynomial>(new Polynomial);
  for (auto& keyx : x->m_data)
    if(keyx.first->length()<maxLength)
      for(auto& keyy : y->m_data)
	if(keyy.first->length()<maxLength){
	  auto t = productLyndonWords(s,*keyx.first,*keyy.first,maxLength);
	  if(t){
	    auto scalar = productCoefficients(keyx.second, keyy.second);
	    for(auto& key : t->m_data)
	      key.second = productCoefficients(key.second, scalar);
	    sumPolynomials(s,*out,*t);
	  }
	}
  return out;
}
//returns 0 or a new Polynomial
std::unique_ptr<Polynomial> productLyndonWords(WordPool& s, const LyndonWord& a, const LyndonWord& b, int maxLength, bool check){
  if(check){
    int alen = a.length(), blen=b.length();
    if (alen+blen>maxLength)
      return nullptr;
    if (&a==&b || (alen==blen && a.isEqual(b)))
      return nullptr;
    if(s.lexicographicLess(&b,&a)){
      auto x = productLyndonWords(s,b,a,maxLength,false);
      if(x){
	for(auto& i : x->m_data)
	  for(auto& j : i.second.m_details)
	    j.second *= -1;
      }
      return x;
    }
  }
  auto candidate = s.newLyndonWord(a,b);//don't know yet if this is a LW, we might be able to save creating this,
  if(s.lexicographicLess(candidate,&b) && (a.isLetter() || !s.lexicographicLess(a.getRight(),&b))){
    return polynomialOfWord(candidate);
  }
  auto a1 = productPolynomials(s,polynomialOfWord(a.getRight()).get(), productLyndonWords(s,b,*a.getLeft(),maxLength ).get(),maxLength);
  auto a2 = productPolynomials(s,polynomialOfWord(a.getLeft()).get() , productLyndonWords(s,*a.getRight(),b,maxLength).get(),maxLength);
  if(!a1)
    return a2;
  if(a2)
    sumPolynomials(s,*a1,*a2);
  return a1;
}

void printListOfLyndonWords(int d, int m){
  WordPool s;
  auto list = makeListOfLyndonWords(s,d,m);
  for (const LyndonWord* l : list){
    printLyndonWordBracketsDigits(*l, std::cout);
    std::cout<<"\n";
  }
}

Coefficient basicCoeff(int i){
  Input inp;
  inp.m_index = i;
  Coefficient out;
  out.m_details.resize(1);
  out.m_details[0].first.push_back(inp);
  out.m_details[0].second = 1;
  return out;  
}

Polynomial bch(WordPool& s, std::unique_ptr<Polynomial> x, std::unique_ptr<Polynomial> y, int m){
  if(m>20)
    throw std::runtime_error("Coefficients only available up to level 20");
  /*
  std::cout<<"x:";
  printPolynomial(*x,std::cout);
  std::cout<<"y:";
  printPolynomial(*y,std::cout);
  */
  auto bchTable = ReadBCH::read();
  std::vector<std::unique_ptr<Polynomial>> arr;
  Polynomial out = *x; //deep except the LWs
  Polynomial yCopy = *y;
  sumPolynomials(s,out,yCopy);
  arr.reserve(bchTable.m_totalLengths[m-1]);
  arr.push_back(std::move(x));
  arr.push_back(std::move(y));
  for(int i=2; i<bchTable.m_totalLengths[m-1];++i){
    const auto& row = bchTable.m_rows[i];
    arr.push_back(productPolynomials(s,arr[row.m_left-1].get(),arr[row.m_right-1].get(),m));
  }
  
  //printPolynomial(out,std::cout);
  /*
  int fpp = 0;
  for(auto& i : arr){
    std::cout<<fpp++<<": ";
    printPolynomial(*i, std::cout);
    std::cout<<"\n";
    } 
  */
		    
  for(int i=2; i<bchTable.m_totalLengths[m-1];++i){
    const auto& row = bchTable.m_rows[i];
    auto& p = arr[i];
    for(auto& m : p->m_data)
      for(auto& c : m.second.m_details)
	c.second *= row.m_coeff;
    sumPolynomials(s,out,*p);
  }
  return out;
}

void calcFla(int d, int m){
  WordPool s;
  auto wordList = makeListOfLyndonWords(s,d,m);
  std::sort(wordList.begin(),wordList.end(),[&s](LyndonWord* a, LyndonWord* b){
      return s.lexicographicLess(a,b);});
  std::unique_ptr<Polynomial> lhs(new Polynomial);
  std::unique_ptr<Polynomial> rhs(new Polynomial);
  for(int i=0; i<(int)wordList.size(); ++i){
    lhs->m_data.push_back(std::make_pair(wordList[i],basicCoeff( i+1)));
    rhs->m_data.push_back(std::make_pair(wordList[i],basicCoeff(-i-1)));
  }
  std::cout<<"bchbefore"<<std::endl;
  auto poly = bch(s,std::move(lhs),std::move(rhs),m);
  std::cout<<"bchdone"<<std::endl;
  printPolynomial(poly,std::cout);
}
#endif
