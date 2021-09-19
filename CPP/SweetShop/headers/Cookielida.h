
#ifndef COOKIELIDA_H
#define COOKIELIDA_H
#include"Icecream.h"
#include"Cookie.h"
#define Fee 0.1
class Cookielida :public Icecream , public Cookie
{
public:
	Cookielida(const Taste& icecream, const Cookies& cookie);//c'tor that get icecream and cookie and make a cookielida with 2 of the cookies and 1 icecream ball
	virtual float GetPrice();//return the final price of a cookielida
	inline Taste& GetCookielidaIcecream()//return the iceicream taste of the cookielida
	{
		return GetTaste();
	}
	inline Cookies& GetCookielidaCookies()//return the cookie taste of the cookilida
	{
		return GetCookie();
	}
	inline float CancellationFee() //return the cancellationfee that the customer have to pay if he canceled an order of cookielida
	{
		return GetPrice() * Fee;
	}

};
#endif