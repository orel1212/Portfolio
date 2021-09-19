
#ifndef CUSTOMER_H
#define CUSTOMER_H
#include"Cookielida.h"
#include"Candy.h"
#include "String.h"
#include<typeinfo>
#define NoCust 0
#define DefaultSize 1
typedef enum { Candiess = 1, Cookiess, Icecreamss, Cookielidass } Sweet;
class Customer
{
	String name;
	int id;
	Sweetitem ** array;
	int size;
public:
	Customer(const String& name, const int& id);//c'tor that get name and id and make new customer obj
	Sweetitem * operator [] (const int& Index) const;//operator [] that return the sweetitem* in the index
	inline Sweetitem ** GetArray() //return the whole sweetitem ** array
	{
		return this->array;
	}
	inline int GetSize() const//return the size of the array
	{
		return this->size;
	}
	inline int GetID() const//return the id of the customer
	{
		return this->id;
	}
	Customer& operator+=(Sweetitem *);//operator += that get sweetitem* and add it to the sweetitem ** array
	void RemoveItem(const int& index, float& CancellationFee);//get index and remove the item in the index, the cancellationfee is just to return by reference the cancellation fee if the deleted item is cookielida
	bool operator!=(Sweetitem *);//operator !+ that get sweetitem* and check if there is same sweetitem in the array
	void PrintSweetArray();//print the sweet array to the screen
	inline String& GetName() //return the name of the customer
	{
		return this->name;
	}
	~Customer();//d'tor
	
};
#endif