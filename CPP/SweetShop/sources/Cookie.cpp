
#include"Cookie.h"
Cookie::Cookie()
{}
Cookie::Cookie(const Cookies& cookie, const int& numofcookies, const float& price) :Sweetitem(), cookie(cookie), numofcookies(numofcookies), price(price)
{}
float Cookie::GetPrice()
{
	return this->price * this->numofcookies;
}