
#ifndef SWEETSHOP_H
#define SWEETSHOP_H
#include"Customer.h"
#include<windows.h>
#include<iomanip>
#define Gray 8
#define Blue 9
#define Green 10
#define Aqua 11
#define RED 12
#define Purple 13
#define Yellow 14
#define WHITE 15
#define MemberPrice 15
#define Percentage 100
#define Discount 0.05
typedef enum{IsCandy=0,IsCookielida,IsCookie,IsIcecream} HighestBought;
#define TempSize 4
class SweetShop
{
	Customer ** customer;
	int size;
	void AddCustomerToMember(const String& name);//get name and add the name to the file
	bool CheckIfAMember(const String& name);//get name to check if the name is already in the file
	void OrderMenu(const int& index);//get index of customer and manage all his orders
	Customer * operator [] (const int& Index) const;//operator [] that return the customer * in the index
	void FindTheBiggestBuyer();//function that find the biggest buyer in the customer array
	void FindWhichTypeIsMostBought(const int& index,const float& finalbuy);//fucntion that check which type of sweetitem is most bought in the biggest buyer,get index of the biggest buyer and his amount of money he paid
	void PrintToFile(const float& totalrevenue,const String& date);//get the date and the revenue of this day and print it to the file
	void AddSweetToArray(const int& index, const Candies& candies);//get index of the customer and candy and add it to the sweetitem ** array
	void AddSweetToArray(const int& index, const Cookies& cookies);//get index of the customer and cookie and add it to the sweetitem ** array
	void AddSweetToArray(const int& index, const Taste& icecreams);//get index of the customer and icecream and add it to the sweetitem ** array
	void AddSweetToArray(const int& index, const Taste& icecreams, const Cookies& cookies);//get index of the customer and cookie and icecream and add it to the sweetitem ** array
public:
	SweetShop();//c'tor
	void StartDay(const String& date);//the function that manage all the system, get date
	~SweetShop(); //d'tor
};
#endif