#ifndef READ_BCH_COEFFS_H
#define READ_BCH_COEFFS_H

#include<fstream>
#include<iostream>
#include<numeric>
#include<stdexcept>
#include<vector>

//#define BCHFILE "bchLyndon20.dat" //set this in setup.py, Manifest.in
//#define BCHFILE "bchHall20.dat" //set this in setup.py, Manifest.in

#include "bchLyndon20.dat.h"

namespace ReadBCH{

  struct Row{
    int m_left, m_right;
    double m_coeff;
  };

  struct FileData{
    std::vector<Row> m_rows;
    std::vector<int> m_totalLengths;
  };
  
  FileData read(){
    /*
    std::ifstream is(BCHFILE);
    if(!is.is_open())
      throw std::runtime_error("Can't open file " BCHFILE);
    */
    std::istringstream is(bchLyndon20_dat);
    FileData o;
    std::vector<Row>& out = o.m_rows;
    std::vector<int> n_ofEachOrder(20,0);
    n_ofEachOrder[0]=2;
    std::vector<int> orders (2,1);
    orders.reserve(115000);
    out.reserve(115000);//approximate
    std::string s;
    int nrow, left, right;
    double numerator, denominator; //if these are int64_t, you finish about half way
    while(is>>nrow>>left>>right>>numerator>>denominator){
      Row r;
      r.m_left = left;
      r.m_right = right;
      r.m_coeff = ((double)numerator)/denominator;
      out.push_back(r);
      if(nrow>2){
	int order = orders[left-1]+orders[right-1];
	orders.push_back(order);
	++n_ofEachOrder[order-1];
      }
    }
    std::partial_sum(n_ofEachOrder.begin(),n_ofEachOrder.end(),std::back_inserter(o.m_totalLengths));
    //std::cout<<nrow<<"\n";
    return o;
  }
};

#endif
