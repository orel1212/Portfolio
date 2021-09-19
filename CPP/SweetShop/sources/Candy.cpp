
#include"Candy.h"
Candy::Candy()
{
}
Candy::Candy(const Candies& candy, const float& weight, const float& price) :Sweetitem(), candy(candy), weighting(weight), priceinkg(price)
{
}
float Candy::GetPrice()
{
	return this->priceinkg * this->weighting * GtoKG;
}
