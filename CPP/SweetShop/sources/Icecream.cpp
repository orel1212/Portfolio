
#include"Icecream.h"
Icecream::Icecream()
{}
Icecream::Icecream(const Taste& icecream, const int& numofballs, const float& price) :Sweetitem(), icecream(icecream), numofballs(numofballs), price(price)
{}
float Icecream::GetPrice()
{
	return this->price * this->numofballs;
}