#pragma once
#include <assert.h>
#include <iostream>
#include <algorithm>

template <typename Vector>
class VectorIterator
{
public:
	using ValueType = typename Vector::value_type;
	using PtrType = ValueType*;
	using RefType = ValueType&;
private:
	PtrType m_vptr;
	size_t m_size;
public:
	VectorIterator(PtrType v_ptr,size_t size): m_vptr(v_ptr), m_size(size)
	{}
	VectorIterator& operator++()
	{
		m_vptr++;
		return *this;
	}
	VectorIterator operator++(int)
	{
		VectorIterator vi = *this;
		++(*this);
		return vi;
	}
	VectorIterator& operator--()
	{
		m_vptr--;
		return *this;
	}
	VectorIterator operator--(int)
	{
		VectorIterator vi = *this;
		--(*this);
		return vi;
	}
	RefType operator[](size_t idx)
	{
		if (m_size >0) //avoid crashing the program when using iterator end
			assert(idx < m_size);
		return *(m_vptr + idx);
	}
	RefType operator*()
	{
		return *m_vptr;
	}
	PtrType operator->()
	{
		return m_vptr;
	}
	bool operator==(const VectorIterator& other) const
	{
		return m_vptr == other.m_vptr;
	}
	bool operator!=(const VectorIterator& other) const
	{
		return m_vptr != other.m_vptr;
	}
};

template <typename T>
class Vector
{
public:
	using value_type = T;
	using Iterator = VectorIterator<Vector<T>>;
private:
	T* m_data = nullptr;
	const size_t m_new_size_mult = 2;
	size_t m_size = 0;
	size_t m_capacity = 0; //buffer to avoid so many reallocation
	void ReAllocateMem(size_t new_capacity)
	{
		if (new_capacity < 1)
			return;
		T* new_data = (T*)::operator new(new_capacity * sizeof(T)); // avoid c'toring
		size_t size_to_cpy = std::min(new_capacity, m_size);
		for (size_t i = 0; i < size_to_cpy; i++)
		{
			new(&new_data[i]) T(std::move(m_data[i]));// if move c'tor is available, use it instead of copy c'tor
		}
		ClearMem();
		m_size = size_to_cpy;
		m_data = new_data;
		m_capacity = new_capacity;
	}
	void ClearMem()
	{
		for (size_t i = 0; i < m_size; i++)
		{
			m_data[i].~T();
		}
		::operator delete(m_data, m_capacity * sizeof(T));
	}

public:
	Vector()
	{
	}
	Vector(const Vector<T>& other) = delete;
	~Vector()
	{
		ClearMem();
	}
	friend std::ostream& operator<<(std::ostream& os, const Vector<T>& vec)
	{
		if (vec.Size() == 0)
			os << "The current vector is empty...!\n";
		else
		{
			os << "Printing the current vector...!\n";
			for (size_t i = 0; i < vec.Size(); i++)
			{
				os << vec[i] << "\n";
			}
		}
		os << "--------\n";
		return os;
	}
	void PopBack()
	{
		if (m_size > 0)
		{
			m_data[--m_size].~T();
		}
	}
	void PushBack(const T& val)
	{
		if (m_size >= m_capacity)
		{
			ReAllocateMem(m_size * m_new_size_mult + 1);
		}
		m_data[m_size++] = val;
	}
	void PushBack(T&& val)
	{
		if (m_size >= m_capacity)
		{
			ReAllocateMem(m_size * m_new_size_mult + 1);
		}
		m_data[m_size++] = std::move(val);
	}
	size_t Size() const { return m_size; }
	T& operator[](size_t idx)
	{
		assert(idx < m_size);
		return m_data[idx];
			
	}
	const T& operator[](size_t idx) const
	{
		assert(idx < m_size);
		return m_data[idx];

	}
	template<typename... Arguments>
	T& EmplaceBack(Arguments&&... args)
	{
		if (m_size >= m_capacity)
		{
			ReAllocateMem(m_size * m_new_size_mult + 1);
		}
		new(&m_data[m_size]) T(std::forward<Arguments>(args)...); // call c'tor direct over heap, inplace
		return m_data[m_size++];
	}
	
	Iterator begin()
	{
		return Iterator(m_data,m_size);
	}
	Iterator end()
	{
		return Iterator(m_data+m_size,0);
	}
};
