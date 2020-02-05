#include "BasicSparse.hpp"
#include <iostream>
#include <complex>

template<class T>
void SparseStructInfo(BasicSparse::SparseStruct<T>& SS);

template<class T>
int SparseStructCheck(BasicSparse::SparseStruct<T>& SS,size_t rows,size_t cols,size_t nz,bool compressed);

template<class T>
bool SparseStructCheck_i(BasicSparse::SparseStruct<T>& SS,size_t idx,size_t val){return !(SS.i[idx]==val);}

template<class T>
bool SparseStructCheck_p(BasicSparse::SparseStruct<T>& SS,size_t idx,size_t val){return !(SS.p[idx]==val);}

template<class T>
bool SparseStructCheck_x(BasicSparse::SparseStruct<T>& SS,size_t idx,T val){return !(SS.x[idx]==val);}

//unit test file
int main(){
  
  BasicSparse::UnVector<int> uv(10,0);
  for (auto i :uv){
    std::cout << i << " "; 
  }
  std::cout << ", capacity = " << uv.capacity() << std::endl;

  int count=0;
  for (auto& i :uv){
    i=++count;
  }
  for (auto i :uv){
    std::cout << i << " "; 
  }
  std::cout << ", capacity = " << uv.capacity() << std::endl;

  uv.push_back(11);
  for (auto i :uv){
    std::cout << i << " "; 
  }
  std::cout << ", capacity = " << uv.capacity() << std::endl;

  BasicSparse::UnVector<int> uv0(0,-2);
  for (auto i :uv0){
    std::cout << i << " "; 
  }
  std::cout << ", capacity = " << uv0.capacity() << std::endl;

  uv0.push_back(1);
  for (auto i :uv0){
    std::cout << i << " "; 
  }
  std::cout << ", capacity = " << uv0.capacity() << std::endl;

  uv0=uv;
  for (auto i :uv0){
    std::cout << i << " "; 
  }
  std::cout << ", capacity = " << uv0.capacity() << std::endl; 
    
  BasicSparse::SparseStruct<int> A1 {2,3,0};
  std::cout << "SparseStruct<int> construction error: " << SparseStructCheck(A1,2,3,0,0) <<std::endl;

  
  BasicSparse::SparseStruct<std::complex<double>> A2 {17,4,3}; //reserve 3, but doesn't lead to size() being 3!
  std::cout << "SparseStruct<std::complex<double>> construction error: " << SparseStructCheck(A2,17,4,0,0) <<std::endl;
  
  BasicSparse::UnVector<BasicSparse::uSpInt> ivec {4,6,3,0,5,5};
  BasicSparse::UnVector<BasicSparse::uSpInt> jvec {2,1,0,2,1,1};
  BasicSparse::UnVector<int > xvec_int{{2,0,1,19,-12,12}};
  BasicSparse::UnVector<std::complex<double> > xvec_cd{{-2.3e-15,std::complex<double>(1.1,3.0),0.0,std::complex<double>(0.0,-0.005),std::complex<double>(4.3,-81.2),std::complex<double>(-4.3,81.2)}};

  BasicSparse::SparseStruct<int> A3 {7,3,ivec,jvec,xvec_int};  
  std::cout << "Preallocated arrays SparseStruct<int> construction error: " << SparseStructCheck(A3,7,3,6,0) <<std::endl;

  BasicSparse::SparseStruct<std::complex<double> > A4 {7,3,ivec,jvec,xvec_cd};  
  std::cout << "Preallocated arrays SparseStruct<std::complex<double>> construction error: " << SparseStructCheck(A4,7,3,6,0) <<std::endl;

  
  std::cout << "Checking i value errors: " << SparseStructCheck_i<std::complex<double> >(A4,2,3) <<std::endl;
  std::cout << "Checking p value errors: " << SparseStructCheck_p<std::complex<double> >(A4,1,1) <<std::endl;
  std::cout << "Checking x value errors: " << SparseStructCheck_x<std::complex<double> >(A4,3,std::complex<double>(0.0,-0.005)) <<std::endl;

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
  std::cout << "Transpose just cols 3 to 5" <<std::endl;
  TransposeCols(A4,3,5).print();
  std::cout << "transpose whole array again (no conjugate)" <<std::endl;
  A4.transpose();
  A4.print();

  std::cout << "Done!" <<std::endl;

  std::cout << "Check norm()" <<std::endl;
  std::cout << A3.norm() <<std::endl;
  std::cout << A4.norm() <<std::endl;
  std::cout << "Done!" <<std::endl;

  std::cout << "Check column M.permute(): swap cols 0 and 2 " <<std::endl;
  A4.permute(BasicSparse::UnVector<BasicSparse::uSpInt>(),BasicSparse::UnVector<BasicSparse::uSpInt>({{2,1,0}}));
  A4.print();

  std::cout << "Check row M.permute(): swap rows 4 and 1 (note row 1 was empty)" <<std::endl;
  A4.permute(BasicSparse::UnVector<BasicSparse::uSpInt>({{0,4,2,3,1,5,6}}),BasicSparse::UnVector<BasicSparse::uSpInt>());
  A4.print();

  std::cout << "Check equals" <<std::endl;
  BasicSparse::SparseStruct<std::complex<double> > A5=A4;
  A5.print();
  
  std::cout << "Check row M.permute(): swap rows 5 and 6 (breaks row ordering!)" <<std::endl;
  A4.permute(BasicSparse::UnVector<BasicSparse::uSpInt>({{0,1,2,3,1,6,5}}),BasicSparse::UnVector<BasicSparse::uSpInt>());
  A4.print();
  
  std::cout << "Check M.drop(1.0e-10)" <<std::endl;
  A4.drop(1.0e-10);
  A4.print();

  std::cout << "Check addition M - conj(M)" <<std::endl;
  Add(std::complex<double>(1.0),A4,std::complex<double>(-1.0),A4,0,1).print();

  std::cout << "Check M+=M" <<std::endl;
  A4+=A4;
  A4.print();

  std::cout << "Check M-=M" <<std::endl;
  A4-=A4;
  A4.print();

  std::cout << "Check ArrayCatByRow: [M M]" <<std::endl;
  A4.x[0]=std::complex<double>(-2.0,0.11);
  A4.x[1]=std::complex<double>(5,0.0);
  A4.print();

  for (auto p : A4.p){std::cout << p << " ";}
  std::cout << std::endl;
  
  BasicSparse::ArrayCatByRow(A4,A4).print();

  for (auto p : A4.p){std::cout << p << " ";}
  std::cout << std::endl;

  A5.transpose().print();

  Transpose(A5).print();
  
  BasicSparse::UnVector<BasicSparse::uSpInt> nzs1(A5.rows);
  BasicSparse::MultiplySubArrayNonZeros(nzs1,A5,Transpose(A5),0,2);
  
  for (auto n : nzs1){
    std::cout << n << " ";
  }
  std::cout << std::endl;

  BasicSparse::UnVector<BasicSparse::uSpInt> nzs2(A5.cols);
  BasicSparse::MultiplySubArrayNonZeros(nzs2,Transpose(A5),A5,0,6);
  
  for (auto n : nzs2){
    std::cout << n << " ";
  }
  std::cout << std::endl;

  BasicSparse::Multiply({1.0,0.0},A5,{1.0,0.0},Transpose(A5),0,0).print();


  
  BasicSparse::Multiply({1.0,0.0},Transpose(A5),{1.0,0.0},A5,0,0).print();
  
  return 0;
}

template<class T>
void SparseStructInfo(BasicSparse::SparseStruct<T>& SS){
  std::cout << "SparseStructInfo" << std::endl;
  std::cout <<SS.rows <<", " << SS.cols << ", " << SS.nonzeros() << ", " << SS.compressed() << std::endl ;
}

template<class T>
int SparseStructCheck(BasicSparse::SparseStruct<T>& SS,size_t rows,size_t cols,size_t nz,bool compressed){
  
  if (rows==SS.rows
      && cols==SS.cols
      && nz == SS.nonzeros()
      && compressed == SS.compressed()){
    if (!compressed && SS.i.size()!=SS.p.size()){
      return -1;
    }
    return 0;
  }
  else{
    SparseStructInfo(SS);
  }
    return 1;
}
