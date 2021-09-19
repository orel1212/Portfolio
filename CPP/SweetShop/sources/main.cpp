
#include "SweetShop.h"
int main()
{
	String day;
	cout << "Welcome to the sweet shop!" << endl;
	cout << "Please enter the start date in DD/MM/YYYY format" << endl;
	cin >> day;
	SweetShop sweetShop;
	sweetShop.StartDay(day);
}