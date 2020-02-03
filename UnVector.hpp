/** @file UnVector.hpp 
 * A container class similar to std vector, but not as clever, and with a simple array as the underlying storage
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
    T* storage_;
    std::size_t capacity_;

    T* allocate_(std::size_t n){
      if (n) {capacity_=n;return new T[n];}
      else {capacity_=0;return nullptr;}
    }

    void reallocate_(std::size_t new_capacity){
      T* new_storage=allocate_(new_capacity);
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

    UnVector() : size_(0),storage_(allocate_(2)){};
    UnVector(std::size_t n) : size_(n), storage_(allocate_(n)){};
    UnVector(std::size_t n, const T& val) : UnVector(n) {
      for (auto i=0; i<n; ++i){
	storage_[i]=val;
      }
    };
    UnVector(const std::initializer_list<T>& l) : UnVector(l.size()){
      std::size_t idx=0;
      for (auto& elem : l){
	storage_[idx++]=elem;
      }
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
      return *(storage_-1);
    }

    const T& back() const {
      return *(storage_-1);
    }
    
    void push_back(const T& val){
      if (size_==capacity_) reallocate_(capacity_ ? 2* capacity_ : 2);
      storage_[size_++]=val;
    }

    void resize(std::size_t s){
      if (s > capacity_) reallocate_(s);
      size_=s;
    };

    UnVector<T>& append(const UnVector<T>& to_append){
      if (capacity_< size_+to_append.size_) reallocate_(size_+to_append.size_);
      for (std::size_t i=0;i< to_append.size_; ++i){
	storage_[size_+i]=to_append[i];
      }
      size_+=to_append.size_;
      return *this;
    }

    UnVector<T>& append(const T* begin_ptr, std::size_t num){
      if (capacity_< size_+num) reallocate_(size_+num);
      for (std::size_t i=0;i< num; ++i){
	storage_[size_+i]=*(begin_ptr+i);
      }
      size_+=num;
      return *this;
    }
    
    T* begin(){return storage_;}
    T* end(){return &(storage_[size_]);}

    const T* begin() const {return storage_;}
    const T* end() const {return &(storage_[size_]);}

    std::size_t size() const {return size_;}
    std::size_t capacity() const {return capacity_;}
    
  };

  /*template <typename T, typename A=std::allocator<T>>
  class default_init_allocator : public A {
    typedef std::allocator_traits<A> a_t;
  public:
    template <typename U> struct rebind {
      using other =
	default_init_allocator<
        U, typename a_t::template rebind_alloc<U>
	>;
    };
    
    using A::A;
    
    template <typename U>
    void construct(U* ptr)
      noexcept(std::is_nothrow_default_constructible<U>::value) {
      ::new(static_cast<void*>(ptr)) U;
    }
    template <typename U, typename...Args>
    void construct(U* ptr, Args&&... args) {
      a_t::construct(static_cast<A&>(*this),
		     ptr, std::forward<Args>(args)...);
    }
  };

  template<class T>
  using UnVector=std::vector<T,default_init_allocator<T> >;
  */
}
