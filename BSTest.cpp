#include "BasicSparse.hpp"
#include <iostream>
#include <complex>

template<class T>
void SparseStorageInfo(BasicSparse::SparseStorage<T>& SS);

template<class T>
int SparseStorageCheck(BasicSparse::SparseStorage<T>& SS,size_t rows,size_t cols,size_t nz,bool compressed);

template<class T>
bool SparseStorageCheck_i(BasicSparse::SparseStorage<T>& SS,size_t idx,size_t val){return !(SS.i.at(idx)==val);}

template<class T>
bool SparseStorageCheck_p(BasicSparse::SparseStorage<T>& SS,size_t idx,size_t val){return !(SS.p.at(idx)==val);}

template<class T>
bool SparseStorageCheck_x(BasicSparse::SparseStorage<T>& SS,size_t idx,T val){return !(SS.x.at(idx)==val);}

//unit test file
int main(int argc, char** argv){

  BasicSparse::SparseStorage<int> A1 {2,3,0};
  std::cout << "SparseStorage<int> construction error: " << SparseStorageCheck(A1,2,3,0,0) <<std::endl;

  
  BasicSparse::SparseStorage<std::complex<double>> A2 {17,4,3}; //reserve 3, but doesn't lead to size() being 3!
  std::cout << "SparseStorage<std::complex<double>> construction error: " << SparseStorageCheck(A2,17,4,0,0) <<std::endl;

  std::vector<BasicSparse::uSpInt> ivec {{4,6,3,0,5,5}};
  std::vector<BasicSparse::uSpInt> jvec {{2,1,0,2,1,1}};
  std::vector<int > xvec_int{{2,0,1,19,-12,12}};
  std::vector<std::complex<double> > xvec_cd{{-2.3e-15,std::complex<double>(1.1,3.0),0.0,std::complex<double>(0.0,-0.005),std::complex<double>(4.3,-81.2),std::complex<double>(-4.3,81.2)}};

  BasicSparse::SparseStorage<int> A3 {7,3,ivec,jvec,xvec_int};  
  std::cout << "Preallocated arrays SparseStorage<int> construction error: " << SparseStorageCheck(A3,7,3,6,0) <<std::endl;

  BasicSparse::SparseStorage<std::complex<double> > A4 {7,3,ivec,jvec,xvec_cd};  
  std::cout << "Preallocated arrays SparseStorage<std::complex<double>> construction error: " << SparseStorageCheck(A4,7,3,6,0) <<std::endl;

  
  std::cout << "Checking i value error: " << SparseStorageCheck_i<std::complex<double> >(A4,2,3) <<std::endl;
  std::cout << "Checking p value error: " << SparseStorageCheck_p<std::complex<double> >(A4,1,1) <<std::endl;
  std::cout << "Checking x value errors: " << SparseStorageCheck_x<std::complex<double> >(A4,3,std::complex<double>(0.0,-0.005)) <<std::endl;

  std::cout <<"integer array test" <<std::endl;
  std::cout << "print triplet" <<std::endl;
  A3.print();
  std::cout << "print compressed" <<std::endl;
  A3.compress();
  A3.print();
  std::cout << "sum duplicate entries" <<std::endl;
  A3.sum_duplicates();
  A3.print();
  std::cout << "transpose" <<std::endl;
  A3.transpose();
  A3.print();
  std::cout << "transpose back" <<std::endl;
  A3.transpose();
  A3.print();
  
  std::cout << "Done!" <<std::endl;

  std::cout <<"complex<double> test" <<std::endl;
  std::cout << "print triplet" <<std::endl;
  A4.print();
  std::cout << "print compressed" <<std::endl;
  A4.compress();
  A4.print();
  std::cout << "sum duplicate entries" <<std::endl;
  A4.sum_duplicates();
  A4.print();
  std::cout << "transpose conjugate" <<std::endl;
  A4.transpose(1);
  A4.print();
  std::cout << "transpose again (no conjugate)" <<std::endl;
  A4.transpose();
  A4.print();
  
  std::cout << "Done!" <<std::endl;

  std::cout << "Check norm()" <<std::endl;
  std::cout << A3.norm() <<std::endl;
  std::cout << A4.norm() <<std::endl;
  std::cout << "Done!" <<std::endl;

  std::cout << "Check column permute(): swap cols 0 and 2 " <<std::endl;
  A4.permute(std::vector<BasicSparse::uSpInt>(),std::vector<BasicSparse::uSpInt>({{2,1,0}}));
  A4.print();

  std::cout << "Check row permute(): swap rows 4 and 1 (note row 1 was empty)" <<std::endl;
  A4.permute(std::vector<BasicSparse::uSpInt>({{0,4,2,3,1,5,6}}),std::vector<BasicSparse::uSpInt>());
  A4.print();

  std::cout << "Check row permute(): swap rows 5 and 6 (breaks row ordering!)" <<std::endl;
  A4.permute(std::vector<BasicSparse::uSpInt>({{0,1,2,3,1,6,5}}),std::vector<BasicSparse::uSpInt>());
  A4.print();

  std::cout << "Check drop(1.0e-10)" <<std::endl;
  A4.drop(1.0e-10);
  A4.print();

  std::cout << "Check addition" <<std::endl;
  add(std::complex<double>(1.0),A4,std::complex<double>(-1.0),A4,0,1).print();
  
  return 0;
}

template<class T>
void SparseStorageInfo(BasicSparse::SparseStorage<T>& SS){
  std::cout << "SparseStorageInfo" << std::endl;
  std::cout <<SS.rows <<", " << SS.cols << ", " << SS.nonzeros() << ", " << SS.compressed << std::endl ;
}

template<class T>
int SparseStorageCheck(BasicSparse::SparseStorage<T>& SS,size_t rows,size_t cols,size_t nz,bool compressed){
  
  if (rows==SS.rows && cols==SS.cols && nz == SS.nonzeros() && compressed == SS.compressed){
    if (!compressed && SS.i.size()!=SS.p.size()){
      return -1;
    }
    return 0;
  }
  else return 1;
}
