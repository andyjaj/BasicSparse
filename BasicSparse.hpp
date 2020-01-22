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
#endif

namespace BasicSparse {

  typedef unsigned long long int uSpInt;
  typedef long long int SpInt; //should be at least 64 bit

  template <class T>
  class SparseStorage;
  
  template <class T>
  SparseStorage<T> add(T alpha, const SparseStorage<T>& A, T beta, const SparseStorage<T>& B, bool conjA=0, bool conjB=0);

  template <class T>
  uSpInt scatter(const SparseStorage<T>& A, uSpInt j, T beta, std::vector<uSpInt>& w, std::vector<T>& x, uSpInt mark, SparseStorage<T>& C, uSpInt nz, bool conjugate);

  template<class T>
  T conj(T val); 
  
  ///////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////
  
  template <class T>
  class SparseStorage {
  public:
    
    //
    //variables
    //
    
    uSpInt rows;
    uSpInt cols;
    std::vector<uSpInt> i; //row indices
    std::vector<uSpInt> p; //col indices (if triplet), col ptrs (if compressed)
    std::vector<T> x; // the actual values in the array
    bool compressed; //is the array compressed sparse column (true), or is it in triplet form (false)

    //
    // constructors
    //
    
    //constructor for uncompressed version
    
    SparseStorage(uSpInt r, uSpInt c, uSpInt nz_reserve=0) : rows(r), cols(c) { //construct an empty triplet
      i.reserve(nz_reserve); p.reserve(nz_reserve); x.reserve(nz_reserve); //get the memory (if known) to avoid lots of reallocs
    }

    //constructor from existing data
    SparseStorage(uSpInt r, uSpInt c, const std::vector<uSpInt>& ivec, const std::vector<uSpInt>& pvec, const std::vector<T>& xvec, bool cmp=0) : rows(r), cols(c), i(ivec), p(pvec), x(xvec), compressed(cmp) {} 
    SparseStorage(uSpInt r, uSpInt c, std::vector<uSpInt>&& ivec, std::vector<uSpInt>&& pvec, std::vector<T>&& xvec, bool cmp=0) : rows(r), cols(c), i(std::move(ivec)), p(std::move(pvec)), x(std::move(xvec)), compressed(cmp) {} 

    //
    // member functions
    //
    
    uSpInt nonzeros() const {return i.size();} //should always match the number of nonzero row entries

    SparseStorage<T>& compress();
    SparseStorage<T>& sum_duplicates();
    SparseStorage<T>& transpose(bool conjugate=0);
    SparseStorage<T>& permute(const std::vector<uSpInt>& new_rows, const std::vector<uSpInt>& new_col);
    SparseStorage<T>& drop(T tol = T{}); //drops values with abs val smaller than tol
    
    void entry(uSpInt r, uSpInt c, T val); //make an entry in triplet form
    
    double norm() const;
    void print() const;

    friend SparseStorage<T> add<T>(T alpha, const SparseStorage<T>& A, T beta, const SparseStorage<T>& B, bool conjA, bool conjB);
        
  };

  //out of class definitions
  
  template <class U>  
  SparseStorage<U>& SparseStorage<U>::compress(){
    //compress a triplet storage array
    //this will not order the rows, but necessarily orders the columns
    if (compressed) {
      //Compressing an already compressed array?
      abort();
    }
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
    compressed=1;
    i=ci;
    p=cp;
    x=cx;
    
    return *this;
  }
  
  template <class U>  
  SparseStorage<U>& SparseStorage<U>::sum_duplicates(){
    if (!compressed) compress();
    
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
    if (!compressed) compress();
    
    std::vector<uSpInt> w(rows,0); //workspace of size rows
    
    for (uSpInt col_ptr=0; col_ptr<p[cols]; ++col_ptr) w[i[col_ptr]]++; //count number in each row
    std::vector<uSpInt> cp(rows+1); //compressed col ptrs
    std::partial_sum(w.begin(),w.end(),cp.begin()+1); //populate array of column pointers
    cp[0]=0; //(1st element should be zero)
    std::copy(cp.begin(),cp.begin()+rows,w.begin()); //copy the [0 to (rows-1)] row totals back into w, which will be used to update the positions in each (new) column
    
    std::vector<uSpInt> ci(nonzeros()); //compressed col ptrs
    std::vector<U> cx(nonzeros()); //compressed col ptrs
    
    for (uSpInt j = 0; j < cols; ++j){ //loop over cols
      for (uSpInt col_ptr = p[j]; col_ptr < p[j+1]; ++col_ptr){//go through non zero elements in col
	uSpInt q = w[i[col_ptr]]++; //new (transposed) column pointer
	ci[q]=j; ///new row is old column
	//if type is complex, conjugate if necessary
	cx[q]= conjugate ? cx[q]=conj(x[col_ptr]) : cx[q]=x[col_ptr]; 
      }
    }
    
    //update this array
    std::swap(rows,cols);
    i=ci;
    p=cp;
    x=cx;
    
    return *this;
  }

  template <class U>  
  SparseStorage<U>& SparseStorage<U>::permute(const std::vector<uSpInt>& new_rows_inv, const std::vector<uSpInt>& new_cols){
    if (!compressed) compress();

    uSpInt nz=0;
    
    std::vector<uSpInt> ci(nonzeros());
    std::vector<uSpInt> cp(cols+1);
    std::vector<U> cx(nonzeros());

    for (uSpInt n = 0; n < cols; ++n){ //loop through all cols
      cp[n] = nz; // column n of new is column new_cols[n] of original
      uSpInt j = new_cols.size() ? new_cols[n] : n;
      for (uSpInt t = p[j]; t < p[j+1]; ++t)
        {
	  cx[nz] = x[t];  /* row i of original is row new_rows_inv[i] of new */
	  ci[nz++] = new_rows_inv.size() ? (new_rows_inv[i[t]]) : i[t];
        }
    }
    cp[cols] = nz; //update final col pointer

    i=ci;
    p=cp;
    x=cx;
    
    return *this;
  }

  template <class U>  
  SparseStorage<U>& SparseStorage<U>::drop(U tol){ //drops values with abs val smaller than tol
    if (!compressed) compress();

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
  void SparseStorage<U>::print() const {
    std::cout << std::endl << "Sparse array output" <<std::endl;
    std::cout << nonzeros() << " nonzero elements in " << rows << " by " << cols << " ";
    if (compressed) {
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
    if (!compressed) {
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
  SparseStorage<U> add(U alpha, const SparseStorage<U>& A, U beta, const SparseStorage<U>& B, bool conjA, bool conjB){

    if (!A.compressed || !B.compressed){
      std::cerr << "Trying to add arrays not in csc form " << A.compressed << " " << B.compressed << std::endl;
      abort();
    }
    
    if (A.rows != B.rows || A.cols != B.cols) {
      std::cerr << "Trying to add arrays of different shape " << A.rows << " by " << A.cols << " and " << B.rows << " by " << B.cols << std::endl;
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
      nz=scatter(A,col,alpha,w,x,col+1,C,nz,conjA); //updates nz and populates
      nz=scatter(B,col,beta,w,x,col+1,C,nz,conjB); //updates nz and populates
      for(uSpInt idx = C.p[col] ; idx < nz ; ++idx){
	C.x[idx] = x[C.i[idx]];//use working x vector to populate C.x[]
      }
    }
    //we really want the answer to be row ordered to avoid surprises
    C.p[cols]=nz;// update final new nz
    C.i.resize(nz);
    C.x.resize(nz);
    
    return C;
  }

  template <class U>
  uSpInt scatter(const SparseStorage<U>& A, uSpInt j, U beta, std::vector<uSpInt>& w, std::vector<U>& x, uSpInt mark, SparseStorage<U>& C, uSpInt nz, bool conjugate){
    if (!A.compressed){
      std::cerr << "Trying to scatter for array not in csc form" << std::endl;
      abort();
    }
    for (uSpInt idx=A.p[j]; idx<A.p[j+1];++idx){
      uSpInt row = A.i[idx];
      if (w[row]<mark){
	w[row]=mark; //newentry for this row in col j
	C.i[nz++]=row;
	x[row]=beta* (conjugate ? conj(A.x[idx]) : A.x[idx]);
      }
      else x[row]+=beta*(conjugate ? conj(A.x[idx]) : A.x[idx]);
    }
    return nz;
  }

  template<class U>
  U conj(U val) {return val;}
  
  template<>
  inline std::complex<double> conj(std::complex<double> val) {
    return std::conj(val);
  }
  
}

#endif
