
#ifndef COOKIE_H
#define COOKIE_H
#include"Sweetitem.h"
typedef enum { Chocolate = 1, Chocolatechips , Vanilla , Coffee } Cookies;
class Cookie :virtual public Sweetitem
{
	Cookies cookie;
	int numofcookies;
	float price;
public:
	virtual float GetPrice();//return the final price of a cookie
	Cookie();
	Cookie(const Cookies& cookie, const int& numofcookies, const float& price);//c'tor that get a cookie , numof cookies and a price and make a new cookie obj
	inline Cookies& GetCookie()//return the cookie
	{
		return this->cookie;
	}
	inline int GetNumOfCookies()//return the num of cookies that the customer wanted
	{
		return this->numofcookies;
	}
};
#endif