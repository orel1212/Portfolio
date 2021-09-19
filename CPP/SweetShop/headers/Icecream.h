
#ifndef ICECREAM_H
#define ICECREAM_H
#include"Sweetitem.h"
typedef enum { ItalianChocolate = 1, BulgerianChocolatechips, GermanVanilla, RomanianCoffee } Taste;
class Icecream :virtual public Sweetitem
{
	Taste icecream;
	int numofballs;
	float price;
public:
	virtual float GetPrice();//return the final price of the icecream
	Icecream();
	Icecream(const Taste& icecream, const int& numofballs, const float& price);//get icecream taste,numof balls and price and make new icecream obj
	inline Taste& GetTaste()//return the taste of the cookielida
	{
		return this->icecream;
	}
	inline int GetNumOfBalls()//return the num of balls
	{
		return this->numofballs;
	}
};
#endif