
#ifndef CANDY_H
#define CANDY_H
#include"Sweetitem.h"
#define GtoKG 0.001
typedef enum {Chockolate=1,Lolipop,Snake,Teddybear} Candies;
class Candy : public Sweetitem
{
	Candies candy;
	float weighting;
	float priceinkg;
public:
	virtual float GetPrice();//return the final price of the candy
	Candy();
	Candy(const Candies& candy, const float& weight, const float& price);//c'tor that get a candy,weight and price and make a new candy obj
	inline Candies& GetCandy()//return the candy
	{
		return this->candy;
	}
	inline float GetWeight()//return the weight in grams
	{
		return this->weighting;
	}
};
#endif