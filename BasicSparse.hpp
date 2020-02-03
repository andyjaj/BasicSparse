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
  class SparseStruct;

  template <class T>
  SparseStruct<T> Transpose(const SparseStruct<T>& A, bool conjugate=0);

  template <class T>
  SparseStruct<T> TransposeCols(const SparseStruct<T>& A, uSpInt FirstCol, uSpInt LastCol, bool conjugate=0);
  
  template <class T>
  SparseStruct<T> Permute(const SparseStruct<T>& A, const std::vector<uSpInt>& new_rows_inv, const std::vector<uSpInt>& new_cols);  
  
  template <class T>
  SparseStruct<T> Add(T alpha, const SparseStruct<T>& A, T beta, const SparseStruct<T>& B, bool conjA=0, bool conjB=0);

  template <class T>
  SparseStruct<T> Multiply(T alpha, const SparseStruct<T>& A, T beta, const SparseStruct<T>& B, bool conjA=0, bool conjB=0);

  template <class T>
  std::vector<uSpInt> MultiplySubArrayNonZeros(const SparseStruct<T>& L,const SparseStruct<T>& R,uSpInt FirstRcol,uSpInt LastRcol);
  
  template <class T>
  SparseStruct<T> MultiplySubArray(T alpha, const SparseStruct<T>& A, T beta, const SparseStruct<T>& B,uSpInt FirstBcol,uSpInt LastBcol, const std::vector<uSpInt>& nzs, bool conjA=0, bool conjB=0);

  template <class T>
  SparseStruct<T>& ArrayCatByRow(SparseStruct<T>& L,const SparseStruct<T>& R);
  
  template <class T>
  uSpInt Scatter(const SparseStruct<T>& A, uSpInt j, T beta, std::vector<uSpInt>& w, std::vector<T>& x, uSpInt mark, SparseStruct<T>& C, uSpInt nz, bool conjugate);

  template <class T>
  uSpInt SimpleScatter(const SparseStruct<T>& L, const SparseStruct<T>& R, uSpInt j, std::vector<uSpInt>& w);
  
  template<class T>
  T Conj(T val);
  
  template<class U>
  std::complex<U> Conj(std::complex<U> val); //overload (with a template param)
  
  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////
  
  template <class T>
  class SparseStruct {
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
    SparseStruct(uSpInt r, uSpInt c, uSpInt nz_reserve=0) : rows(r), cols(c) { //construct an empty triplet
      i.reserve(nz_reserve); p.reserve(nz_reserve); x.reserve(nz_reserve); //get the memory (if known) to avoid lots of reallocs
    }
    //constructor from existing data
    SparseStruct(uSpInt r, uSpInt c, const std::vector<uSpInt>& ivec, const std::vector<uSpInt>& pvec, const std::vector<T>& xvec, bool cmp=0) : rows(r), cols(c), i(ivec), p(pvec), x(xvec), compressed_(cmp) {} 
    SparseStruct(uSpInt r, uSpInt c, std::vector<uSpInt>&& ivec, std::vector<uSpInt>&& pvec, std::vector<T>&& xvec, bool cmp=0) : rows(r), cols(c), i(std::move(ivec)), p(std::move(pvec)), x(std::move(xvec)), compressed_(cmp) {} 
    //
    // member functions
    //  
    uSpInt nonzeros() const {return i.size();} //should always match the number of nonzero row entries
    bool compressed() const {return compressed_;}
    bool valid() const;
    
    SparseStruct<T>& compress();
    SparseStruct<T>& sum_duplicates();
    SparseStruct<T>& transpose(bool conjugate=0);
    SparseStruct<T>& permute(const std::vector<uSpInt>& new_rows, const std::vector<uSpInt>& new_col);
    SparseStruct<T>& drop(T tol = T{}); //drops values with abs val smaller than tol
    SparseStruct<T>& operator=(SparseStruct<T> RHS);
    SparseStruct<T>& operator+=(SparseStruct<T> RHS);
    SparseStruct<T>& operator-=(SparseStruct<T> RHS);
    
    void entry(uSpInt r, uSpInt c, T val); //make an entry in triplet form
    
    double norm() const;
    void print() const;

  private:
    void swap(SparseStruct<T>& RHS);
    
  };

  //
  //Out-of-class member definitions
  //

  template <class U>  
  bool SparseStruct<U>::valid() const {
    if (compressed_ && i.size()==x.size() && p.size()==cols+1 && p.back()==nonzeros()) return 1;
    else if (i.size()==p.size() && p.size()==x.size()) return 1;

    return 0;
  }
  
  template <class U>  
  SparseStruct<U>& SparseStruct<U>::compress(){
    //compress a triplet storage array
    //this will not order the rows, but necessarily orders the columns
    
    if (!valid() || compressed()) abort(); //if invalid or already compressed abort
    
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
  SparseStruct<U>& SparseStruct<U>::sum_duplicates(){
    if (!valid() || !compressed()) abort();
    
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


  SparseStruct<U>& SparseStruct<U>::transpose(bool conjugate){
    SparseStruct<U> ans(Transpose(*this,conjugate));
    swap(ans);
    return *this;
  }

  template <class U>  
  SparseStruct<U>& SparseStruct<U>::permute(const std::vector<uSpInt>& new_rows_inv, const std::vector<uSpInt>& new_cols){
    SparseStruct<U> ans(Permute(*this,new_rows_inv,new_cols));
    swap(ans);
    return *this;
  }

  template <class U>  
  SparseStruct<U>& SparseStruct<U>::drop(U tol){ //drops values with abs val smaller than tol
    if (!valid() || !compressed()) abort();

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
  SparseStruct<U>& SparseStruct<U>::operator=(SparseStruct<U> RHS){
    swap(RHS);
    return *this;
  }

  template <class U>
  SparseStruct<U>& SparseStruct<U>::operator+=(SparseStruct<U> RHS){
    *this=Add(U(1.0),*this,U(1.0),RHS);
    return *this;
  }

  template <class U>
  SparseStruct<U>& SparseStruct<U>::operator-=(SparseStruct<U> RHS){
    *this=Add(U(1.0),*this,U(-1.0),RHS);
    return *this;
  }
  
  template <class U>  
  void SparseStruct<U>::print() const {
    std::cout << std::endl << "Sparse array output" <<std::endl;
    std::cout << nonzeros() << " nonzero elements in " << rows << " by " << cols << " ";
    if (valid()){
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
    }
    else std::cout << "Invalid array" <<std::endl;
    
    std::cout << std::endl;
  }

  template <class U>  
  void SparseStruct<U>::entry(uSpInt r, uSpInt c, U val){
    if (!compressed()) {
      i.push_back(r);p.push_back(c);x.push_back(val);
    }
    else
      abort();
  } //make an entry in triplet form

  template <class U>  
  double SparseStruct<U>::norm() const {
    if (!compressed() || !valid()) abort();
    double ans{};
    for (uSpInt col = 0 ; col < cols ; ++col){
      double s{}; //zero init
      for (uSpInt cp = p[col]; cp < p[col+1]; ++cp) s += std::abs(x[cp]) ;
      ans  = s > ans ? s : ans;
    }
    return (ans) ;
  }

  template <class U>
  void SparseStruct<U>::swap(SparseStruct<U>& RHS){
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
  SparseStruct<U> Transpose(const SparseStruct<U>& A, bool conjugate){
    return TransposeCols(A, 0, A.cols-1, conjugate);
  }

  template <class U> //Transpose a subarray defined by cols
  SparseStruct<U> TransposeCols(const SparseStruct<U>& A, uSpInt FirstCol, uSpInt LastCol, bool conjugate){
    if (!A.compressed() || !A.valid()) abort();
    if (FirstCol>LastCol || LastCol > A.cols-1){std::cerr << "Invalid cols for transpose" <<std::endl; abort();}
    
    std::vector<uSpInt> w(A.rows,0); //workspace of size rows
    
    for (uSpInt idx=A.p[FirstCol]; idx<A.p[LastCol+1]; ++idx) w[A.i[idx]]++; //count number in each row
    std::vector<uSpInt> cp(A.rows+1); //compressed col ptrs
    std::partial_sum(w.begin(),w.end(),cp.begin()+1); //populate array of column pointers
    cp[0]=0; //(1st element should be zero)
    std::copy(cp.begin(),cp.begin()+A.rows,w.begin()); //copy the [0 to (rows-1)] row totals back into w, which will be used to update the positions in each (new) column
    
    std::vector<uSpInt> ci(A.nonzeros()); //compressed col ptrs
    std::vector<U> cx(A.nonzeros()); //compressed col ptrs
    
    for (uSpInt j = FirstCol; j < LastCol+1; ++j){ //loop over cols
      for (uSpInt col_ptr = A.p[j]; col_ptr < A.p[j+1]; ++col_ptr){//go through non zero elements in col
	uSpInt q = w[A.i[col_ptr]]++; //new (transposed) column pointer
	ci[q]=j-FirstCol; ///new row is old column (with offset if start col isn't 0)
	//if type is complex, conjugate if necessary
	cx[q]= conjugate ? cx[q]=Conj(A.x[col_ptr]) : cx[q]=A.x[col_ptr]; 
      }
    }
    
    //update this array
    return SparseStruct<U>(LastCol-FirstCol+1,A.rows,std::move(ci),std::move(cp),std::move(cx),A.compressed());
  }

  template <class U>
  SparseStruct<U> Permute(const SparseStruct<U>& A, const std::vector<uSpInt>& new_rows_inv, const std::vector<uSpInt>& new_cols){
    if (!A.compressed() || !A.valid()) abort();

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

    return SparseStruct<U>(A.rows,A.cols,std::move(ci),std::move(cp),std::move(cx),A.compressed());
  }
 
  
  template <class U>
  SparseStruct<U> Add(U alpha, const SparseStruct<U>& A, U beta, const SparseStruct<U>& B, bool conjA, bool conjB){

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

    SparseStruct<U> C(rows,cols,std::vector<uSpInt>(A.nonzeros()+B.nonzeros()),std::vector<uSpInt>(cols+1),std::vector<U>(A.nonzeros()+B.nonzeros()),1);

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
  SparseStruct<U> Multiply(U alpha, const SparseStruct<U>& A, U beta, const SparseStruct<U>& B, bool conjA, bool conjB){
    //this needs a possible threaded version - maybe in a separate include file
    //do some stuff
    
    //ascertain the number of nonzero elements in the answer
    std::vector<uSpInt> nzs=MultiplySubArrayNonZeros(A,B,0,B.cols-1); //vector of new col ptrs, if variant 0
    uSpInt ans_nz=std::accumulate(nzs.begin(),nzs.end(),0);
    //decide on strategy  (B^T A^T)^T or ((A B)^T)^T
    //For 1st case threaded version, we want to avoid doing an extra transpose of B for each thread!

    bool variant=0;

    
    if (ans_nz > A.nonzeros()+B.nonzeros()){std::cout << "Using (B^T A^T)^T" << std::endl;
      variant=1;
      SparseStruct<U> L=Transpose(B);
      SparseStruct<U> R=TransposeCols(A,0,A.cols-1);
      nzs=MultiplySubArrayNonZeros(L,R,0,R.cols-1);
      nzs.push_back(ans_nz);
      //do multiply
      
      return MultiplySubArray(beta, L, alpha, R, 0, A.cols-1, nzs, conjB, conjA).transpose();
    }
    else {std::cout << "Using ((A B)^T)^T" << std::endl;
      const SparseStruct<U>& L=A;
      SparseStruct<U> R=TransposeCols(B,0,B.cols-1);
      nzs.push_back(ans_nz);
      //do multiply
      return MultiplySubArray(alpha, A, beta, B, 0, B.cols-1, nzs, conjA, conjB).transpose().transpose();
    }

  }

  template <class U>
  std::vector<uSpInt> MultiplySubArrayNonZeros(const SparseStruct<U>& L,const SparseStruct<U>& R,uSpInt FirstRcol,uSpInt LastRcol){
    //check cols match rows
    if (!L.compressed() || !R.compressed() || !L.valid() || !R.valid()) abort();
    if (L.cols!=R.rows){std::cout << "Trying to multiply arrays with differing dimensions!" << std::endl; abort();}
    if (FirstRcol > LastRcol || LastRcol > R.cols - 1 ){std::cout << "Requested sub matrix column outside bounds!" << std::endl; abort();}

    std::vector<uSpInt> ans;
    ans.reserve(LastRcol-FirstRcol + 1);

    std::vector<uSpInt> w(L.rows,0);

    for (uSpInt Rcol=FirstRcol;Rcol<LastRcol+1;++Rcol){
      //simplified version of scatter that only calculates the nonzeros for each col
      ans.push_back(SimpleScatter(L,R,Rcol,w));
    }
    return ans;
  }
  
  template <class U>
  SparseStruct<U>  MultiplySubArray(U ml, const SparseStruct<U>& L, U mr, const SparseStruct<U>& R,uSpInt FirstRcol,uSpInt LastRcol, const std::vector<uSpInt>& nzs, bool conjL, bool conjR){

    uSpInt ans_nz=nzs[LastRcol+1]-nzs[FirstRcol];
    /*
    std::vector<uSpInt> i(ans_nz);
    std::vector<uSpInt> p;
    std::vector<U> x(ans_nz);
    */

    
    //std::cout << ans_nz << " nonzeros for ans versus " << A.nonzeros() << " + " << B.nonzeros() << std::endl;

    
    return SparseStruct<U>(L.rows,LastRcol-FirstRcol+1,std::vector<uSpInt>(),std::vector<uSpInt>(LastRcol-FirstRcol+2,0),std::vector<U>(),1);
  }
  
  template <class U>
  SparseStruct<U>& ArrayCatByRow(SparseStruct<U>& L,const SparseStruct<U>& R){
    //take two arrays with the same number of rows, and join them so that cols= L.cols+R.cols
    if (!L.compressed() || !R.compressed()) {std::cerr << "Matrices for row concatenation are not compressed!" << std::endl; abort();}
    if (L.rows!=R.rows) {std::cerr << "Trying to join arrays with differing numbers of rows!" << std::endl; abort();}
    
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
  uSpInt Scatter(const SparseStruct<U>& A, uSpInt j, U beta, std::vector<uSpInt>& w, std::vector<U>& x, uSpInt mark, SparseStruct<U>& C, uSpInt nz, bool conjugate){
    if(!A.compressed()) {std::cerr << "Trying to scatter for array not in csc form" << std::endl; abort();}

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
  inline SparseStruct<U> operator+(SparseStruct<U> lhs, const SparseStruct<U>& rhs)
  {
    lhs += rhs;
    return lhs;
  }

  template<class U>
  inline SparseStruct<U> operator-(SparseStruct<U> lhs, const SparseStruct<U>& rhs)
  {
    lhs -= rhs;
    return lhs;
  }

  template <class U>
  uSpInt SimpleScatter(const SparseStruct<U>& L, const SparseStruct<U>& R, uSpInt j, std::vector<uSpInt>& w){
    if(!L.compressed() || !R.compressed()) {std::cerr << "Trying to Scatter for array not in csc form" << std::endl; abort();}
    if (L.cols!=R.rows){std::cerr << "Trying to multiply arrays with differing dimensions!" << std::endl; abort();}
    if (j > R.cols - 1 ){std::cerr << "Requested column outside bounds!" << std::endl; abort();}
    if (w.size()!= L.rows){std::cerr << "Work array is of incorrect length" << std::endl; abort();}
    
    //simplified version of scatter that only calculates the nonzeros for each col of the final answer
    uSpInt Rcol=j;
    uSpInt nz=0;
    for (uSpInt Ridx=R.p[Rcol];Ridx<R.p[Rcol+1];++Ridx){
      uSpInt Lcol=R.i[Ridx];
      for (uSpInt Lidx=L.p[Lcol];Lidx<L.p[Lcol+1];++Lidx){
	if (w[L.i[Lidx]]!=Rcol+1){ //first time for this row/col combination
	  w[L.i[Lidx]]=Rcol+1;
	  nz++;
	}
      }
    }
    return nz;
  }
  
}

#endif
