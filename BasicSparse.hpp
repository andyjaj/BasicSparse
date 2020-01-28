/** @file BasicSparse.hpp 
 * Simple sparse array library
 * Supports triplet and compressed sparse column (csc) formats
 * Can convert from triplet to csc, print, sum up duplicates, and (conjugate) transpose
 * Heavily based on the algorithms in Tim Davis' CSparse example code
 */
#ifndef BasicSparse_H
#define BasicSparse_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <complex>
#include <type_traits>
#include <iostream>

#if defined(USETBB) //if we want to use intel TBB
#include <tbb/tbb.h>
#include <TBB_Multiply.hpp>
#endif

namespace BasicSparse {

  typedef unsigned long long int uSpInt;
  typedef long long int SpInt; //should be at least 64 bit

  template <class T>
  class SparseStorage;

  template <class T>
  SparseStorage<T> Transpose(const SparseStorage<T>& A, bool conjugate=0);

  template <class T>
  SparseStorage<T> Permute(const SparseStorage<T>& A, const std::vector<uSpInt>& new_rows_inv, const std::vector<uSpInt>& new_cols);  
  
  template <class T>
  SparseStorage<T> Add(T alpha, const SparseStorage<T>& A, T beta, const SparseStorage<T>& B, bool conjA=0, bool conjB=0);

  template <class T>
  SparseStorage<T> Multiply(T alpha, const SparseStorage<T>& A, T beta, const SparseStorage<T>& B, bool conjA=0, bool conjB=0);

  template <class T>
  std::vector<uSpInt> MultiplySubArrayNonZeros(const SparseStorage<T>& L,const SparseStorage<T>& R,uSpInt FirstRcol,uSpInt LastRcol);
  
  template <class T>
  SparseStorage<T> SubArrayMultiply(const SparseStorage<T>& L,const SparseStorage<T>& R,uSpInt FirstRcol,uSpInt LastRcol);

  template <class T>
  SparseStorage<T>& ArrayCatByRow(SparseStorage<T>& L,const SparseStorage<T>& R);
  
  template <class T>
  uSpInt Scatter(const SparseStorage<T>& A, uSpInt j, T beta, std::vector<uSpInt>& w, std::vector<T>& x, uSpInt mark, SparseStorage<T>& C, uSpInt nz, bool conjugate);

  template<class T>
  T Conj(T val);
  
  template<class U>
  std::complex<U> Conj(std::complex<U> val); //overload (with a template param)
  
  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////
  
  template <class T>
  class SparseStorage {
    //
    //variables
    //
  public:        
    uSpInt rows;
    uSpInt cols;
    std::vector<uSpInt> i; //row indices
    std::vector<uSpInt> p; //col indices (if triplet), col ptrs (if compressed)
    std::vector<T> x; // the actual values in the array
  private:
    bool compressed_; //is the array compressed sparse column (true), or is it in triplet form (false)

  public:        
    //constructor for uncompressed version
    SparseStorage(uSpInt r, uSpInt c, uSpInt nz_reserve=0) : rows(r), cols(c) { //construct an empty triplet
      i.reserve(nz_reserve); p.reserve(nz_reserve); x.reserve(nz_reserve); //get the memory (if known) to avoid lots of reallocs
    }
    //constructor from existing data
    SparseStorage(uSpInt r, uSpInt c, const std::vector<uSpInt>& ivec, const std::vector<uSpInt>& pvec, const std::vector<T>& xvec, bool cmp=0) : rows(r), cols(c), i(ivec), p(pvec), x(xvec), compressed_(cmp) {} 
    SparseStorage(uSpInt r, uSpInt c, std::vector<uSpInt>&& ivec, std::vector<uSpInt>&& pvec, std::vector<T>&& xvec, bool cmp=0) : rows(r), cols(c), i(std::move(ivec)), p(std::move(pvec)), x(std::move(xvec)), compressed_(cmp) {} 
    //
    // member functions
    //  
    uSpInt nonzeros() const {return i.size();} //should always match the number of nonzero row entries
    bool compressed() const {return compressed_;}
    
    SparseStorage<T>& compress();
    SparseStorage<T>& sum_duplicates();
    SparseStorage<T>& transpose(bool conjugate=0);
    SparseStorage<T>& permute(const std::vector<uSpInt>& new_rows, const std::vector<uSpInt>& new_col);
    SparseStorage<T>& drop(T tol = T{}); //drops values with abs val smaller than tol
    SparseStorage<T>& operator=(SparseStorage<T> RHS);
    SparseStorage<T>& operator+=(SparseStorage<T> RHS);
    SparseStorage<T>& operator-=(SparseStorage<T> RHS);
    
    void entry(uSpInt r, uSpInt c, T val); //make an entry in triplet form
    
    double norm() const;
    void print() const;

  private:
    void swap(SparseStorage<T>& RHS);
    
  };

  //
  //Out-of-class member definitions
  //
  
  template <class U>  
  SparseStorage<U>& SparseStorage<U>::compress(){
    //compress a triplet storage array
    //this will not order the rows, but necessarily orders the columns
    
    if (compressed()) abort(); //if already compressed abort
    
    //workspace
    std::vector<uSpInt> w(cols,0);
    
    for (uSpInt n = 0; n < nonzeros(); ++n) w[p[n]]++;           //record number of entries in each col
    std::vector<uSpInt> cp(cols+1); //compressed col ptrs - always at least 2 in size
    std::partial_sum(w.begin(),w.end(),cp.begin()+1); //populate array of column pointers
    cp[0]=0; //create array of column pointers (1st element should be zero)
    std::copy(cp.begin(),cp.begin()+cols,w.begin()); //copy the [0 to (cols-1)] col ptrs back into w, which will be used to update the positions in each column
    
    std::vector<uSpInt> ci(nonzeros()); //compressed row indices
    std::vector<U> cx(nonzeros()); //compressed values	
    
    for (uSpInt n = 0; n < nonzeros(); ++n){ //for each nonzero elem
      uSpInt col_ptr=w [p[n]]++; //get column position for this element
      ci [col_ptr] = i[n] ;    //record i and x for this element
      cx[col_ptr] = x[n] ;
    }
    
    //update this array to compressed version
    compressed_=1;
    i=ci;
    p=cp;
    x=cx;
    
    return *this;
  }
  
  template <class U>  
  SparseStorage<U>& SparseStorage<U>::sum_duplicates(){
    if (!compressed()) compress();
    
    //make a workspace, the size of the number of rows
    std::vector<SpInt> w(rows,-1);
    uSpInt nz=0;
    for (uSpInt j = 0 ; j < cols; ++j){//go through cls
      uSpInt q = nz; //q is the new col_ptr for this col
      for (uSpInt loop_p=p[j];loop_p<p[j+1];++loop_p){ //go through elements in this column
	uSpInt loop_i=i[loop_p]; //row idx for element
	//need to compare unsigned integer with integer
	if (w[loop_i]>=0 && (uSpInt) w[loop_i]>=q){ //if w val is greater than col_ptr, we have seen this element before IN THIS COLUMN
	  x[w[loop_i]]+=x[loop_p]; //if yes, sum up
	}
	else { //haven't seen this element before
	  w[loop_i] = nz; //most recent occurrence of row in the li
	  i[nz] = loop_i; //keep value at i,j
	  x[nz++] = x[loop_p]; //and increment the number of nonzeros 
	}
      }
      p[j]=q;
    }
    p[cols]=nz; //new number of nonzeros
    
    //resize vectors to new nz size
    i.resize(nz);// this also updates the number of nonzeros
    x.resize(nz);
    
    return *this;
  }

  template <class U>  


  SparseStorage<U>& SparseStorage<U>::transpose(bool conjugate){
    SparseStorage<U> ans(Transpose(*this,conjugate));
    swap(ans);
    return *this;
  }

  template <class U>  
  SparseStorage<U>& SparseStorage<U>::permute(const std::vector<uSpInt>& new_rows_inv, const std::vector<uSpInt>& new_cols){
    SparseStorage<U> ans(Permute(*this,new_rows_inv,new_cols));
    swap(ans);
    return *this;
  }

  template <class U>  
  SparseStorage<U>& SparseStorage<U>::drop(U tol){ //drops values with abs val smaller than tol
    if (!compressed()) compress();

    double abstol=std::abs(tol);
    uSpInt nz=0;
    for (uSpInt col = 0; col < cols; ++col){
        uSpInt col_ptr = p[col];                        /* get current location of col j */
        p[col] = nz;                       /* record new location of col j */
        for (uSpInt idx=col_ptr; idx < p[col+1]; ++idx){
	  if (std::abs(x[idx])>abstol){
	    x[nz] = x[idx];  /* keep A(i,j) */
	    i[nz++] = i[idx];
	  }
        }
    }
    p[cols] = nz;
    i.resize(nz);// this also updates the number of nonzeros
    x.resize(nz);
    return *this;
  }

  template <class U>
  SparseStorage<U>& SparseStorage<U>::operator=(SparseStorage<U> RHS){
    swap(RHS);
    return *this;
  }

  template <class U>
  SparseStorage<U>& SparseStorage<U>::operator+=(SparseStorage<U> RHS){
    *this=Add(U(1.0),*this,U(1.0),RHS);
    return *this;
  }

  template <class U>
  SparseStorage<U>& SparseStorage<U>::operator-=(SparseStorage<U> RHS){
    *this=Add(U(1.0),*this,U(-1.0),RHS);
    return *this;
  }
  
  template <class U>  
  void SparseStorage<U>::print() const {
    std::cout << std::endl << "Sparse array output" <<std::endl;
    std::cout << nonzeros() << " nonzero elements in " << rows << " by " << cols << " ";
    if (compressed()) {
      std::cout << "csc" <<std::endl;
      std::cout << "idx \t i \t j\t x" <<std::endl;
      for (uSpInt col=0; col<cols;++col){
	for (uSpInt cp=p[col];cp<p[col+1];++cp){
	  std::cout << cp << " \t " << i[cp] << " \t " << col << " \t " << x[cp] <<std::endl;
	}
      }
    }
    else {
      std::cout << "triplet" <<std::endl;
      std::cout << "i \t j\t x" <<std::endl;

      for (uSpInt n=0; n<nonzeros();++n){
	std::cout << i[n] << " \t " << p[n] << " \t " << x[n] <<std::endl;	
      }
    }

    if (!nonzeros())
      std::cout <<"Empty array"<<std::endl;
    
    std::cout << std::endl;
  }

  template <class U>  
  void SparseStorage<U>::entry(uSpInt r, uSpInt c, U val){
    if (!compressed()) {
      i.push_back(r);p.push_back(c);x.push_back(val);
    }
    else
      abort();
  } //make an entry in triplet form

  template <class U>  
  double SparseStorage<U>::norm() const {
    double ans{};
    for (uSpInt col = 0 ; col < cols ; ++col){
      double s{}; //zero init
      for (uSpInt cp = p[col]; cp < p[col+1]; ++cp) s += std::abs(x[cp]) ;
      ans  = s > ans ? s : ans;
    }
    return (ans) ;
  }

  template <class U>
  void SparseStorage<U>::swap(SparseStorage<U>& RHS){
    std::swap(this->rows,RHS.rows);
    std::swap(this->cols,RHS.cols);
    std::swap(this->i,RHS.i);
    std::swap(this->p,RHS.p);
    std::swap(this->x,RHS.x);
    std::swap(this->compressed_,RHS.compressed_);
  }

  //
  //Non-member function defs
  //
   
  template <class U>
  SparseStorage<U> Transpose(const SparseStorage<U>& A, bool conjugate){
    if (!A.compressed()) return SparseStorage<U>(0,0);
    
    std::vector<uSpInt> w(A.rows,0); //workspace of size rows
    
    for (uSpInt idx=0; idx<A.p[A.cols]; ++idx) w[A.i[idx]]++; //count number in each row
    std::vector<uSpInt> cp(A.rows+1); //compressed col ptrs
    std::partial_sum(w.begin(),w.end(),cp.begin()+1); //populate array of column pointers
    cp[0]=0; //(1st element should be zero)
    std::copy(cp.begin(),cp.begin()+A.rows,w.begin()); //copy the [0 to (rows-1)] row totals back into w, which will be used to update the positions in each (new) column
    
    std::vector<uSpInt> ci(A.nonzeros()); //compressed col ptrs
    std::vector<U> cx(A.nonzeros()); //compressed col ptrs
    
    for (uSpInt j = 0; j < A.cols; ++j){ //loop over cols
      for (uSpInt col_ptr = A.p[j]; col_ptr < A.p[j+1]; ++col_ptr){//go through non zero elements in col
	uSpInt q = w[A.i[col_ptr]]++; //new (transposed) column pointer
	ci[q]=j; ///new row is old column
	//if type is complex, conjugate if necessary
	cx[q]= conjugate ? cx[q]=Conj(A.x[col_ptr]) : cx[q]=A.x[col_ptr]; 
      }
    }
    
    //update this array
    return SparseStorage<U>(A.cols,A.rows,std::move(ci),std::move(cp),std::move(cx),A.compressed());
  }

  template <class U>
  SparseStorage<U> Permute(const SparseStorage<U>& A, const std::vector<uSpInt>& new_rows_inv, const std::vector<uSpInt>& new_cols){
    if (!A.compressed()) return SparseStorage<U>(0,0);

    uSpInt nz=0;
    
    std::vector<uSpInt> ci(A.nonzeros());
    std::vector<uSpInt> cp(A.cols+1);
    std::vector<U> cx(A.nonzeros());

    for (uSpInt n = 0; n < A.cols; ++n){ //loop through all cols
      cp[n] = nz; // column n of new is column new_cols[n] of original
      uSpInt j = new_cols.size() ? new_cols[n] : n;
      for (uSpInt t = A.p[j]; t < A.p[j+1]; ++t)
        {
	  cx[nz] = A.x[t];  //row i of original is row new_rows_inv[i] of new
	  ci[nz++] = new_rows_inv.size() ? (new_rows_inv[A.i[t]]) : A.i[t];
        }
    }
    cp[A.cols] = nz; //update final col pointer

    return SparseStorage<U>(A.rows,A.cols,std::move(ci),std::move(cp),std::move(cx),A.compressed());
  }
 
  
  template <class U>
  SparseStorage<U> Add(U alpha, const SparseStorage<U>& A, U beta, const SparseStorage<U>& B, bool conjA, bool conjB){

    if (!A.compressed() || !B.compressed()){
      std::cerr << "Trying to Add arrays not in csc form " << A.compressed() << " " << B.compressed() << std::endl;
      abort();
    }
    
    if (A.rows != B.rows || A.cols != B.cols) {
      std::cerr << "Trying to Add arrays of different shape " << A.rows << " by " << A.cols << " and " << B.rows << " by " << B.cols << std::endl;
      abort();
    }

    uSpInt rows = A.rows;
    uSpInt cols = B.cols;
    
    std::vector<uSpInt> w(rows,0); //used to indicate if row entries exist or not
    std::vector<U> x(rows);

    SparseStorage<U> C(rows,cols,std::vector<uSpInt>(A.nonzeros()+B.nonzeros()),std::vector<uSpInt>(cols+1),std::vector<U>(A.nonzeros()+B.nonzeros()),1);

    uSpInt nz=0;
    for (uSpInt col=0; col<cols; ++col){
      C.p[col]=nz; //col of C starts here
      nz=Scatter(A,col,alpha,w,x,col+1,C,nz,conjA); //updates nz and populates
      nz=Scatter(B,col,beta,w,x,col+1,C,nz,conjB); //updates nz and populates
      for(uSpInt idx = C.p[col] ; idx < nz ; ++idx){
	C.x[idx] = x[C.i[idx]];//use working x vector to populate C.x[]
      }
    }
    C.p[cols]=nz;// update final new nz
    C.i.resize(nz);
    C.x.resize(nz);

    //we really want the answer to be row ordered to avoid surprises
    return C.transpose().transpose();
  }

  template <class U>
  SparseStorage<U> Multiply(U alpha, const SparseStorage<U>& A, U beta, const SparseStorage<U>& B, bool conjA, bool conjB){
    //this needs a possible threaded version - maybe in a separate include file
    //do some stuff

    //ascertain the number of nonzero elements in the answer
    //uSpInt ans_nz=Multiply_nz_count(A,B);
    
    //decide on strategy ((A B)^T)^T or (B^T A^T)^T 
    
    return SparseStorage<U>(A.rows,B.cols,std::vector<uSpInt>(),std::vector<uSpInt>(),std::vector<U>(),1);
  }

  template <class U>
  std::vector<uSpInt> MultiplySubArrayNonZeros(const SparseStorage<U>& L,const SparseStorage<U>& R,uSpInt FirstRcol,uSpInt LastRcol){
    //check cols match rows
    if (L.cols!=R.rows){std::cout << "Trying to multiply arrays with differing dimensions!" << std::endl; abort();}
    if (FirstRcol > LastRcol || LastRcol > R.cols - 1 ){std::cout << "Requested sub matrix column outside bounds!" << std::endl; abort();}

    std::vector<uSpInt> ans;
    ans.reserve(LastRcol-FirstRcol + 1);

    std::vector<uSpInt> w(L.rows,0);

    //simplified version of scatter that on;y calculates the nonzeros for each col of the final answer
    for (uSpInt Rcol=FirstRcol;Rcol<LastRcol+1;++Rcol){
      uSpInt nz=0;
      for (uSpInt Ridx=R.p[Rcol];Ridx<R.p[Rcol+1];++Ridx){
	uSpInt Lcol=R.i[Ridx];
	for (uSpInt Lidx=L.p[Lcol];Lidx<L.p[Lcol+1];++Lidx){
	  if (w[L.i[Lidx]]<=Rcol){ //first time for this row/col combination
	    w[L.i[Lidx]]=Rcol+1;
	    nz++;
	  }
	}
      }
      
      ans.push_back(nz);
    }
    return ans;
  }
  
  /*
  template <class U>
  SparseStorage<U> SubArrayMultiply(const SparseStorage<U>& L,const SparseStorage<U>& R,uSpInt FirstRcol,uSpInt LastRcol);
  */
  
  template <class U>
  SparseStorage<U>& ArrayCatByRow(SparseStorage<U>& L,const SparseStorage<U>& R){
    //take two arrays with the same number of rows, and join them so that cols= L.cols+R.cols

    if (L.rows!=R.rows) {std::cout << "Trying to join arrays with differing numbers of rows!" << std::endl; abort();}
    
    uSpInt oldLnz=L.nonzeros();

    //update p's
    L.p.insert(L.p.end(),R.p.begin()+1,R.p.end());//+1 is vital here
    std::transform(L.p.begin()+L.cols+1,L.p.end(),L.p.begin()+L.cols+1,[=](uSpInt p) { return p+oldLnz; });

    L.cols+=R.cols;
    
    L.i.insert(L.i.end(),R.i.begin(),R.i.end());
    L.x.insert(L.x.end(),R.x.begin(),R.x.end());

    return L;
    
  }
  
  template <class U>
  uSpInt Scatter(const SparseStorage<U>& A, uSpInt j, U beta, std::vector<uSpInt>& w, std::vector<U>& x, uSpInt mark, SparseStorage<U>& C, uSpInt nz, bool conjugate){
    if(!A.compressed()) {std::cerr << "Trying to Scatter for array not in csc form" << std::endl; abort();}

    for (uSpInt idx=A.p[j]; idx<A.p[j+1];++idx){
      uSpInt row = A.i[idx];
      if (w[row]<mark){
	w[row]=mark; //newentry for this row in col j
	C.i[nz++]=row;
	if (x.size()) x[row]=beta* (conjugate ? Conj(A.x[idx]) : A.x[idx]);
      }
      else if (x.size()) x[row]+=beta*(conjugate ? Conj(A.x[idx]) : A.x[idx]);
    }
    return nz;
  }

  template<class U>
  U Conj(U val) {return val;}
  
  template<class U>
  std::complex<U> Conj(std::complex<U> val) {return std::conj(val);}

  template<class U>
  inline SparseStorage<U> operator+(SparseStorage<U> lhs, const SparseStorage<U>& rhs)
  {
    lhs += rhs;
    return lhs;
  }

  template<class U>
  inline SparseStorage<U> operator-(SparseStorage<U> lhs, const SparseStorage<U>& rhs)
  {
    lhs -= rhs;
    return lhs;
  }
  
}

#endif
