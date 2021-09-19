
#ifndef SWEETITEM_H
#define SWEETITEM_H
#define Abstract 0
#define NoItems 0
#define IcecreamPrice 5.5
#define CookiePrice 6.5
#define CandyPrice 3.5
#include<iostream>
using namespace std;
class Sweetitem
{
	static int serialnum;
public:
	Sweetitem();
	virtual float GetPrice() = Abstract;//pure virtual function that will be the menu
	static void AddOneToSNUM();//static function that will add 1 if there is a new sweet item 
	static void ReduceOneFromSNUM();//reduce one sweetitem if one sweet item deleted from the customer order
	inline void NoSweet()//print the num of sweets
	{
		cout << "There are " << this->serialnum << " Sweet items" << endl;
	}
	virtual ~Sweetitem();//virtual d'tor
	
};
#endif