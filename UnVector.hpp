/** @file UnVector.hpp 
 * A container class similar to std vector, but not as clever, and with a simple array as the underlying storage
 * The main purpose it to allow non-sequential writing to an allocated, but not initialised array.
 */
#ifndef UnVector_H
#define UnVector_H

//#include <vector>
#include <iterator>
#include <utility>
#include <initializer_list>

#endif

namespace BasicSparse {

  template <class T>
  class UnVector{
  private:
    std::size_t size_;
    std::size_t capacity_;
    T* storage_;
    
    T* allocate_(){
      if (capacity_) {
	T* ptr= new T[capacity_];
	return ptr;
      }
      else return nullptr;
    }

    void reallocate_(std::size_t new_capacity){
      capacity_=new_capacity;
      T* new_storage=new_capacity ? new T[new_capacity] : nullptr;
      for (std::size_t i=0;i<size_;++i){
	new_storage[i]=storage_[i];
      }
      std::swap(new_storage,storage_);
      delete[] new_storage;
      
    }
    
    void swap(UnVector& Other){
      std::swap(capacity_,Other.capacity_);
      std::swap(size_,Other.size_);
      std::swap(storage_,Other.storage_);
    }
    
  public:

    UnVector() : size_(0),capacity_(2),storage_(allocate_()){}
    
    UnVector(std::size_t n) : size_(n), capacity_(n), storage_(allocate_()){}
    
    UnVector(std::size_t n, const T& val) : UnVector(n) {
      for (std::size_t i=0; i<n; ++i){
	storage_[i]=val;
      }
    }
    
    UnVector(const std::initializer_list<T>& l) : UnVector(l.size()){
      std::size_t idx=0;
      for (auto& elem : l){
	storage_[idx++]=elem;
      }
    }

    //copy constructor
    UnVector(const UnVector<T>& other) : size_(other.size_), capacity_(other.capacity_),storage_(capacity_ ? new T[capacity_] : nullptr) {
      std::copy(other.storage_,other.storage_+size_,storage_);
    }

    //move constructor
    UnVector(UnVector<T>&& other) noexcept : UnVector<T>() {
      swap(other);
    }
    
    ~UnVector(){
      if (storage_) delete[] storage_;
      storage_=nullptr;
    }


    T& operator[](size_t i){
      return storage_[i];
    }

    const T& operator[](size_t i) const {
      return storage_[i];
    }

    UnVector<T>& operator=(UnVector<T> RHS){
      swap(RHS);
      return *this;
    }
    
    T& back(){
      return storage_[size_-1];
    }

    const T& back() const {
      return storage_[size_-1];
    }
    
    void push_back(const T& val){
      if (size_==capacity_) reallocate_(capacity_ ? 2* capacity_ : 2);
      storage_[size_++]=val;
    }

    void resize(std::size_t s){
      if (s > capacity_) reallocate_(s);
      size_=s;
    };

    UnVector<T>& append(const T* begin_ptr, std::size_t num){

      if (capacity_ < size_+num){       //use temp buffer, in case we are appending to itself
	T* buffer = new T[size_+num];
	for (std::size_t i=0;i<size_;++i) buffer[i]=storage_[i];
	for (std::size_t i=0;i<num; ++i) buffer[size_+i]=begin_ptr[i];
	std::swap(buffer,storage_);
	capacity_=size_+num;
	delete[] buffer;
      }
      else {
	for (std::size_t i=0;i< num; ++i)  storage_[size_+i]=begin_ptr[i];
      }

      size_+=num;
      return *this;
    }

    UnVector<T>& append(const UnVector<T>& to_append){

      return this->append(to_append.begin(),to_append.size());
      
    }
    
    T* begin(){return storage_;}
    T* end(){return storage_ + size_/*&(storage_[size_])*/;}

    const T* begin() const {return storage_;}
    const T* end() const {return storage_ + size_/*&(storage_[size_])*/;}

    std::size_t size() const {return size_;}
    std::size_t capacity() const {return capacity_;}
    
  };
}
