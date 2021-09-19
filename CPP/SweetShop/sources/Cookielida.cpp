
#include"Cookielida.h"
Cookielida::Cookielida(const Taste& icecream, const Cookies& cookie) :Sweetitem(), Icecream(icecream, 1, IcecreamPrice), Cookie(cookie,2,CookiePrice)
{
}
float Cookielida::GetPrice()
{
	return ((CookiePrice * 2 + IcecreamPrice) * 1.5);
}