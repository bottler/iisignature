#ifndef BCH_H
#define BCH_H
#include<algorithm>
#include<functional>
#include<iostream>
#include<iterator>
#include<map>
#include<memory>
#include<utility>
#include<vector>

#include "readBCHCoeffs.hpp"

//If you use StandardHall, calculations will be happen according to 
//the same Hall basis as is used in CoRoPa, and not using the Lyndon word basis.
//Note that in this case, some of the names and comments in this file will be wrong,
// - a LyndonWord is not a Lyndon word but a basis element.
//Everything will still basically work, tests pass etc, 
//but the numbers will be different.

enum class LieBasis {Lyndon, StandardHall};

//using Interrupt = const std::function<void()>&;
typedef void (*Interrupt)();

//amalgamate_adjacent(v.begin(),v.end(),tojoin,amalgamater)
//iterates over v and when finding a range [i..j) for which
//tojoin(*i,*n) for all n in (i,j), replaces it with 
//  (the single element *i if amalgamater(i,j) and nothing otherwise)
//returns one past the last - removes everything made unnecessary

//I.e.: If you have a sequence where you want to merge any sequence of adjacent
//elements which have something in common so that they become
//either a single element or nothing, use this function where
//tojoin identifies elements with the "something in common" and amalgamater makes
//what they become and returns whether they become anything at all.

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
      if(amalgamater(a,temp)){
        if(dest!=a)
          std::iter_swap(dest,a);
        ++dest;
      }
    }
    else if(dest!=a)
      std::iter_swap(dest++,a);
    else
      ++dest;
    a=temp;
  }
  return dest;
}

//amalgamate_adjacent_pairs(v.begin(),v.end(),tojoin,amalgamater)
//iterates over v and  if tojoin(*i,*(i+1)), replaces *i and *(i+1)
//with (the single element *i if amalgamater(*i,*(i+1)) and nothing otherwise)
//returns one past the last - removes everything made unnecessary

//This is the same as the previous function for the case where we know already that
//that there won't be sequences longer than 2 elements with the something in common,
//so we can simplify a bit and we can pass 2 elements instead of a
//pair of iterators into amalgamater.

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
      if(amalgamater(*a,*temp)){
        if(dest!=a)
          std::iter_swap(dest,a);
        ++dest;
      }
      ++temp;
    }
    else if(dest!=a)
      std::iter_swap(dest++,a);
    else
      ++dest;
    a=temp;
  }
  return dest;
}

//3 input version of std::merge - no care about order of equivalent elements, 
//it makes sense not to compare with first1 all the time because in our case
//it will tend to be the shortest.
//This implemention does repeat comparisons unnecessarily - but they are quick in our case.
template<class InputIt1, class InputIt2, class InputIt3, class OutputIt>
OutputIt merge3(InputIt1 first1, InputIt1 last1,
                InputIt2 first2, InputIt2 last2,
                InputIt3 first3, InputIt3 last3,
                OutputIt d_first)
{
  for (; first1 != last1; ++d_first) {
    if (first2 == last2) {
      return std::merge(first1, last1, first3, last3, d_first);
    }
    if (first3 == last3) {
      return std::merge(first1, last1, first2, last2, d_first);
    }
    if (*first2 < *first3) {
      if(*first2 < *first1){
        *d_first = *first2;
        ++first2;
      } else {
        *d_first = *first1;
        ++first1;
      }
    } else {
      if(*first1<*first3){
        *d_first = *first1;
        ++first1;
      } else {
        *d_first = *first3;
        ++first3;
      }
    }
  }
  return std::merge(first2, last2, first3, last3, d_first);
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
    if(isLetter())
      return 1;
    return getLeft()->length() + getRight()->length();
  }
  bool isEqual(const LyndonWord& o) const {
    if(this==&o)
      return true;
    return false;
  }
 private:
  const LyndonWord* m_left; //nullptr if we are a letter
  union{
    const LyndonWord* m_right;
    Letter m_letter;
  };
};

//iterate over letters from left to right
//Beware that if you copy an object of this type (except an end),
//only one of the copies is usable,
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

//Useful when debugging, for e.g. doing extra printing only in a certain case.
//Returns true if the LyndonWord object a is the same string of letters as ss
bool isLyndonWord(const LyndonWord* a, const std::string& ss){
  std::vector<const LyndonWord*> foo;
  LyndonWordIterator x(a, foo);
  LyndonWordIterator end;
  auto y=ss.begin();
  while(x!=end && y!=ss.end()){
    if('1'+(*x)!=*y)
      return false;
    ++x; ++y;
  }
  return x==end  && y==ss.end();  
}

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
  std::map<std::pair<const LyndonWord*,const LyndonWord*>,LyndonWord*> m_products;
  std::vector<std::pair<const LyndonWord*, size_t> > m_orderLookup;
 public:
  static const int eachLength = 2000;
  static const int objectSize = (sizeof(LyndonWord)+sizeof(void*)-1)/sizeof(void*);
  int m_used = eachLength+2;//so we know we need to allocate at the start
  const LieBasis m_basis;
  WordPool(LieBasis basis) : m_basis(basis) {}
  LyndonWord* newLyndonWordFromLetter(Letter l){
    static_assert(2==objectSize, "bad objectSize");
    static_assert(eachLength%objectSize==0, "bad eachLength");
    return new(getSpace()) LyndonWord(l);
  }
  LyndonWord* newLyndonWord(const LyndonWord& left, const LyndonWord& right){
    auto p = std::make_pair(&left,&right);
    LyndonWord*& o = m_products[p];
    if(!o)
      o=new(getSpace()) LyndonWord(left,right);
    return o;
  }
  LyndonWord* concatenateIfAllowed(const LyndonWord& left, const LyndonWord& right){
    auto p = std::make_pair(&left,&right);
    auto i = m_products.find(p);
    if(i==m_products.end())
      return nullptr;
    return i->second;
  }
  void doneAdding(){
    using std::make_pair;
    m_orderLookup.clear();
    std::map<const LyndonWord*,size_t> all;
    for(auto& i : m_products){
      all.insert(make_pair(i.second,0u));
      if(i.first.first->isLetter())
        all.insert(make_pair(i.first.first,0u));
      if(i.first.second->isLetter())
        all.insert(make_pair(i.first.second,0u));
    }
    std::vector<const LyndonWord*> all2;
    all2.reserve(all.size());
    for(auto& i : all)
      all2.push_back(i.first);
    std::sort(all2.begin(), all2.end(), [this](const LyndonWord* l, const LyndonWord* r){
        return lexicographicLess(l,r);});
    for(size_t i=0; i<all2.size(); ++i){
      const LyndonWord* l = all2[i];
      all[l] = i;
    }
    m_orderLookup.reserve(all.size());
    for(auto& i : all)
      m_orderLookup.push_back(i);
  } 
  size_t getProxy(const LyndonWord* w){
    auto lookerUpper = [](const std::pair<const LyndonWord*, size_t>& a, const LyndonWord* b){
      return a.first<b;
    };
    auto a = std::lower_bound(m_orderLookup.begin(), m_orderLookup.end(), w, lookerUpper);
    if (a==m_orderLookup.end() || a->first != w)
      throw std::runtime_error("??");
    return a->second;
  }
  bool /*__attribute__ ((noinline))*/ manualLexicographicLess(const LyndonWord* l, const LyndonWord* r){
    if (m_basis == LieBasis::Lyndon) {
      LyndonWordIterator lit(l, m_spaceForIterator1), rit(r, m_spaceForIterator2), end;
      //  return std::lexicographical_compare(std::ref(lit),std::ref(end),std::ref(rit),std::ref(end));
      return std::lexicographical_compare(lit, end, rit, end);
    }
    else {
      int ll = l->length(), lr = r->length();
      if (ll == lr) {
        if (ll == 1)
          return l->getLetter() < r->getLetter();
        if (manualLexicographicLess(l->getLeft(), r->getLeft()))
          return true;
        if (manualLexicographicLess(r->getLeft(), l->getLeft()))
          return false;
        return manualLexicographicLess(l->getRight(), r->getRight());
      }
      return ll < lr;
    }
  }
  bool lexicographicLess(const LyndonWord* l, const LyndonWord* r){
    if(!m_orderLookup.empty())
      return getProxy(l)<getProxy(r);
    return manualLexicographicLess(l,r);
  }
};

std::vector<std::vector<LyndonWord*>> makeListOfLyndonWords(WordPool& s, int d,int m){
  std::vector<std::vector<LyndonWord*>> words(m);
  words[0].resize(d);
  for(Letter i=0; i<d; ++i){
    words[0][i] = s.newLyndonWordFromLetter(i);
  }
  if(s.m_basis == LieBasis::Lyndon)
    for (int level = 2; level <= m; ++level) {
      for (int leftlength = 1; leftlength<level; ++leftlength) {
        const int rightlength = level - leftlength;
        for (LyndonWord* left : words[leftlength - 1])
          for (LyndonWord* right : words[rightlength - 1])
            if (s.lexicographicLess(left, right) && 
                (left->isLetter() || left->getRight() == right ||
                  !s.lexicographicLess(left->getRight(), right)))
              words[level - 1].push_back(s.newLyndonWord(*left, *right));
      }
    }
  else
    for (int level = 2; level <= m; ++level) {
      for (int leftlength = 1; leftlength<=level/2; ++leftlength) {
        const int rightlength = level - leftlength;
        for (size_t il = 0; il < words[leftlength - 1].size(); ++il) {
          LyndonWord* left = words[leftlength - 1][il];
          for (size_t ir = leftlength < rightlength ? 0 : il + 1;
            ir < words[rightlength - 1].size(); ++ir) {
            LyndonWord* right = words[rightlength - 1][ir];
            if (rightlength==1 || !s.lexicographicLess(left, right->getLeft()))
              words[level - 1].push_back(s.newLyndonWord(*left, *right));
          }
        }
      }
    }
  s.doneAdding();
  return words;
}

//An Input represents an indeterminate number
class Input{
public:
  int m_index;
  bool operator==(const Input& o)const {return m_index==o.m_index;}
  bool operator<(const Input& o)const {return m_index<o.m_index;}
};

//worry later about pooling these allocations

class Coefficient{
public:
  //This represents a sum of a product of a load of inputs and a constant.
  //the vector<Input> is sorted
  std::vector<std::pair<std::vector<Input>,double>> m_details;
};

void printCoefficient(const Coefficient& d, std::ostream& o){
  for(size_t ic = 0; ic<d.m_details.size(); ++ic){
    const auto& c = d.m_details[ic];
    for(const auto& i : c.first){
      o<<"["<<i.m_index<<"]";
    }
    o<<c.second<<"\n";
  }
}

Coefficient productCoefficients (const Coefficient& a, const Coefficient& b){
  Coefficient o;
  auto& out = o.m_details;
  out.reserve(a.m_details.size()*b.m_details.size());
  for(size_t ix = 0; ix<a.m_details.size(); ++ix){
    const auto& x = a.m_details[ix];
    for(size_t iy = 0; iy<b.m_details.size(); ++iy){
      const auto& y = b.m_details[iy];
      out.push_back({});
      out.back().first.resize(x.first.size()+y.first.size());
      std::merge(x.first.begin(),x.first.end(),
                 y.first.begin(),y.first.end(),
                 out.back().first.begin());
        out.back().second = x.second * y.second;
    }
  }
  std::sort(out.begin(), out.end());
  using A = std::pair<std::vector<Input>,double>;
  using I = std::vector<A>::iterator;
  auto i = amalgamate_adjacent(out.begin(),out.end(),
                                [](const A& a, const A& b){return a.first==b.first;},
                                [](I a, I b){
                                  double total = 0;
                                  std::for_each(a,b,[&](const A& a){total += a.second;});
                                  a->second = total;
                                  return std::fabs(total)>0.00000001;});
  out.erase(i,out.end());
  //Todo: we could have a member Coefficient::m_size which was the important bit
  //of m_details, and not erase vectors whose memory 
  //we might want to reuse later. Same with Polynomial
  return o;
}

//a *= b * c
void productCoefficients3 (Coefficient& a, const Coefficient& b, const Coefficient& c){
  Coefficient o;
  auto& out = o.m_details;
  out.reserve(a.m_details.size()*b.m_details.size()*c.m_details.size());
  for(size_t ix = 0; ix<a.m_details.size(); ++ix){
    const auto& x = a.m_details[ix];
    for(size_t iy = 0; iy<b.m_details.size(); ++iy){
      const auto& y = b.m_details[iy];
      for(size_t iz = 0; iz<c.m_details.size(); ++iz){
        const auto& z = c.m_details[iz];
        out.push_back({});
        out.back().first.resize(x.first.size()+y.first.size()+z.first.size());
        merge3(x.first.begin(),x.first.end(),
               y.first.begin(),y.first.end(),
               z.first.begin(),z.first.end(),
               out.back().first.begin());
        out.back().second = x.second * y.second * z.second;
      }
    }
  }
  std::sort(out.begin(), out.end());
  using A = std::pair<std::vector<Input>,double>;
  using I = std::vector<A>::iterator;
  auto i = amalgamate_adjacent(out.begin(),out.end(),
                                [](const A& a, const A& b){return a.first==b.first;},
                                [](I a, I b){
                                  double total = 0;
                                  std::for_each(a,b,[&](const A& a){total += a.second;});
                                  a->second = total;
                                  return std::fabs(total)>0.00000001;});
  out.erase(i,out.end());
  out.swap(a.m_details);
}

//make the lhs be lhs+rhs, no care about rhs
void sumCoefficients(Coefficient& lhs, Coefficient&& rhs){
  auto& a = lhs.m_details;
  auto& b = rhs.m_details;

  size_t ss = a.size();
  a.reserve(a.size()+b.size());
  std::move(b.begin(),b.end(),std::back_inserter(a));
  std::inplace_merge(a.begin(),a.begin()+ss,a.end());

  using A = std::pair<std::vector<Input>,double>;
  a.erase(amalgamate_adjacent_pairs(a.begin(), a.end(),
                                    [](const A& a, const A& b){return a.first==b.first;},
                                    [](A& a, const A& b){
                                      double total = a.second + b.second;
                                      a.second = total;
                                      return std::fabs(total)>0.00000001;
                                    })
          ,a.end());
}

class Polynomial{
public:
  //kept lexicographic within each level
  std::vector<std::vector<std::pair<const LyndonWord*,Coefficient>>> m_data;
};

std::ostream& printPolynomial(Polynomial& poly, std::ostream& o){
  for(auto &l : poly.m_data){
    for(auto& m : l){
      printLyndonWordDigits(*m.first,o);
      o<<" ";
      for(const auto& c : m.second.m_details){
        for(const auto& i : c.first){
          o<<"["<<i.m_index<<"]";
        }
        o<<c.second<<"\n";
      }
    }
  }
  return o;
}

std::unique_ptr<Polynomial> polynomialOfWord(const LyndonWord* w){
    auto a = std::unique_ptr<Polynomial>(new Polynomial);
    auto len = w->length();   
    a->m_data.resize(len);
    auto& dest = a->m_data.back();
    dest.resize(1);
    dest[0].first = w;
    dest[0].second.m_details.resize(1);
    dest[0].second.m_details[0].second=1;
    return a;
}

using Term = std::pair<const LyndonWord*,Coefficient>;

struct TermLess{
  WordPool& m_s;
  TermLess(WordPool& s):m_s(s){}
  //mutable int count = 0;
  bool operator() (const Term& a, const Term& b) const {
  //  ++count;
    return m_s.lexicographicLess(a.first, b.first);
  }
};

void sumPolynomialLevels(WordPool& s, std::vector<std::pair<const LyndonWord*,Coefficient>>& lhs,
    std::vector<std::pair<const LyndonWord*,Coefficient>>& rhs){
  auto comp = TermLess(s);
  auto& a = lhs;
  auto& b = rhs;
  size_t ss = a.size();
  //if b has enough capacity, maybe we should detect and swap them here.
  //also similar in sumCoefficients.
  //or reserve double?
  a.reserve(a.size()+b.size());
  /*
  if(b.capacity()>=a.capacity())
    std::cout<<a.capacity()<<" "<<b.capacity()<<" "<<a.size()<<" "<<b.size()<<std::endl;
  */
  std::move(b.begin(),b.end(),std::back_inserter(a));
  std::inplace_merge(a.begin(),a.begin()+ss,a.end(),std::ref(comp));
  a.erase(amalgamate_adjacent_pairs(a.begin(), a.end(),
                                    [](const Term& a, const Term& b){
                                      return a.first->isEqual(*b.first);},
                                    [](Term& a, Term& b){ 
                                      sumCoefficients(a.second,std::move(b.second)); 
                                      return !a.second.m_details.empty();})
          ,a.end());
}
    
//make the lhs be lhs+rhs, rhs can be pilfered
void sumPolynomials(WordPool& s, Polynomial& lhs, Polynomial& rhs){
  if(lhs.m_data.size()<rhs.m_data.size())
    lhs.m_data.resize(rhs.m_data.size());
  for(size_t l=0; l<rhs.m_data.size(); ++l)
    sumPolynomialLevels(s,lhs.m_data[l],rhs.m_data[l]);
}

std::unique_ptr<Polynomial> 
productLyndonWords(WordPool& s, const LyndonWord& a, const LyndonWord& b, 
                   int maxLength, bool check);

//returns 0 or a new Polynomial
//Does not modify or take ownership of x and y
std::unique_ptr<Polynomial> 
productPolynomials(WordPool& s, const Polynomial* x, const Polynomial* y, int maxLength){
  if(!x || !y){
    return nullptr;
  }
  auto out = std::unique_ptr<Polynomial>(new Polynomial);
  size_t xsize = x->m_data.size(), ysize = y->m_data.size();
  size_t newSize = std::min((size_t)maxLength,xsize+ysize);
  out->m_data.resize(newSize);
  for(size_t xlevel = 1; xlevel<=xsize; ++xlevel){
    for(size_t ylevel = 1; ylevel<=ysize; ++ylevel){
      size_t targetlevel = xlevel+ylevel;
      if(targetlevel>newSize)
        break;
      for (auto& keyx : x->m_data[xlevel-1]){
        for(auto& keyy : y->m_data[ylevel-1]){
          auto t = productLyndonWords(s,*keyx.first,*keyy.first,maxLength, true);
          if(t){
            auto& tt = t->m_data[targetlevel-1];
            for(auto& key : tt)//this loop usually happens not many times I think 
              productCoefficients3(key.second, keyx.second, keyy.second);
            sumPolynomialLevels(s,out->m_data[targetlevel-1],tt);
          }
        }
      }
    }
  }
  return out;
}
//returns 0 or a new Polynomial
std::unique_ptr<Polynomial> 
productLyndonWords(WordPool& s, const LyndonWord& a, const LyndonWord& b, int maxLength, bool check){
  if(check){
    int alen = a.length(), blen=b.length();
    if (alen+blen>maxLength)
      return nullptr;
    if (&a==&b || (alen==blen && a.isEqual(b)))
      return nullptr;
    if(s.lexicographicLess(&b,&a)){
      auto x = productLyndonWords(s,b,a,maxLength,false);
      if(x){
        for(auto& l : x->m_data)
          for(auto&i : l)
            for(auto& j : i.second.m_details)
              j.second *= -1;
      }
      return x;
    }
  }
  auto candidate = s.concatenateIfAllowed(a,b);
  if(candidate)
    return polynomialOfWord(candidate);
  std::unique_ptr<Polynomial> a1, a2;
  if (s.m_basis != LieBasis::Lyndon) {
    a1 = productPolynomials(s, productLyndonWords(s, a, *b.getLeft(), maxLength, true).get(),
      polynomialOfWord(b.getRight()).get(), maxLength);
    a2 = productPolynomials(s, productLyndonWords(s, *b.getRight(), a, maxLength, true).get(),
      polynomialOfWord(b.getLeft()).get(), maxLength);
  }
  else {
    a1 = productPolynomials(s, polynomialOfWord(a.getRight()).get(),
      productLyndonWords(s, b, *a.getLeft(), maxLength, true).get(), maxLength);
    a2 = productPolynomials(s, polynomialOfWord(a.getLeft()).get(),
      productLyndonWords(s, *a.getRight(), b, maxLength, true).get(), maxLength);
  }
  if(!a1)
    return a2;
  if(a2)
    sumPolynomials(s,*a1,*a2);
  return a1;
}

void printListOfLyndonWords(int d, int m){
  WordPool s(LieBasis::Lyndon);
  auto list = makeListOfLyndonWords(s,d,m);
  for(const auto& level : list){
    for (const LyndonWord* l : level){
      printLyndonWordBracketsDigits(*l, std::cout);
      std::cout<<"\n";
    }
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

Polynomial bch(WordPool& s, std::unique_ptr<Polynomial> x, std::unique_ptr<Polynomial> y, 
               int m, Interrupt interrupt){
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
  interrupt();
  arr.reserve(bchTable.m_totalLengths[m-1]);
  arr.push_back(std::move(x));
  arr.push_back(std::move(y));
  for(int i=2; i<bchTable.m_totalLengths[m-1];++i){
    const auto& row = bchTable.m_rows[i];
    arr.push_back(productPolynomials(s,arr[row.m_left-1].get(),arr[row.m_right-1].get(),m));
    interrupt();
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
    for(auto& l : p->m_data)
      for (auto&mm : l)
        for(auto& c : mm.second.m_details)
          c.second *= row.m_coeff; //when m_coeff is zero, we're keeping this in
    sumPolynomials(s,out,*p);
    interrupt();
  }
  return out;
}

void calcFla(int d, int m, Interrupt interrupt){
  WordPool s(LieBasis::Lyndon);
  auto wordList = makeListOfLyndonWords(s,d,m);
  for(auto& l : wordList)
    std::sort(l.begin(),l.end(),[&s](LyndonWord* a, LyndonWord* b){
      return s.lexicographicLess(a,b);});
  std::unique_ptr<Polynomial> lhs(new Polynomial);
  std::unique_ptr<Polynomial> rhs(new Polynomial);
  lhs->m_data.resize(wordList.size());
  rhs->m_data.resize(wordList.size());
  for(int i=0, ii=0; i<(int)wordList.size(); ++i){
    for(size_t j=0; j<wordList[i].size(); ++j,++ii){
      lhs->m_data[i].push_back(std::make_pair(wordList[i][j],basicCoeff( ii+1)));
      rhs->m_data[i].push_back(std::make_pair(wordList[i][j],basicCoeff(-ii-1)));
    }
  }
  
  //printPolynomial(*lhs,std::cout);
  //printPolynomial(*rhs,std::cout);
  //std::cout<<"bchbefore"<<std::endl;
  auto poly = bch(s,std::move(lhs),std::move(rhs),m, interrupt);
  //std::cout<<"bchdone"<<std::endl;
  //printPolynomial(poly,std::cout);
}

#endif
