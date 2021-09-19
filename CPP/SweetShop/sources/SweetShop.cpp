
#include"SweetShop.h"
typedef enum { SweetsMenu = 1, NewMember } Option2;
typedef enum {Pay=1,CancelItem} Option;
SweetShop::SweetShop()
{
	this->customer = new Customer *[DefaultSize];//create new customer ** with size 1
	this->size = 0;//there still no customers
}
void SweetShop::StartDay(const String& date)
{
	int currid=0;
	char isANewMember;
	float totalRevenue=0;
	int option;
	do
	{
		HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hstdout, RED);//make the console output red
		cout << "Are you a new client? click Y-if yes , N - if not" << endl;
		cin >> isANewMember;
		system("cls");//clean the screen
		if (isANewMember <= 'z' && isANewMember >= 'a')
			isANewMember += 'A' - 'a';
		if (isANewMember == 'Y')
		{
			fflush(stdin);
			String name;
			cout << "Please enter your name" << endl;
			cin >> name;
			system("cls");//clean the screen
			Customer * cust = new Customer(name, currid);
			currid++;
			if (this->size == NoCust)//if there's no customers yet so he will be the first
			{
				this->customer[this->size] = cust;
				this->size += 1;
			}
			else//we have to create new array using temp to arrive all the customers before
			{
				Customer ** temp = new Customer *[this->size + 1];
				for (int i = 0; i < this->size; i++)
				{
					temp[i] = this->customer[i];
				}
				temp[this->size] = cust;
				delete[] this->customer;
				this->size += 1;
				this->customer = new Customer *[this->size];
				for (int i = 0; i < this->size; i++)
				{
					this->customer[i] = temp[i];
				}
				delete[] temp;
			}
			OrderMenu(this->size - 1);//we working now on the last customer
			float priceToPay = 0;
			do
			{
				HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
				SetConsoleTextAttribute(hstdout, Green);
				system("cls");//clean the screen
				cout << "What you would like to do: 1-To pay , 2-To cancel an item" << endl;
				cin >> option;
				if (option == Pay)//if the user want to pay
				{
					for (int i = 0; i < this->customer[this->size - 1]->GetSize(); i++)//calcuating the price that the user have to pay
					{
						priceToPay += this->customer[this->size - 1]->GetArray()[i]->GetPrice();
					}
				}
				else if (option == CancelItem)//if the user want to remove some item
				{
					system("cls");//clean the screen
					float cancellationFee = 0;
					int choice;
					cout << "Please choose which item you want to cancel" << endl;
					this->customer[this->size - 1]->PrintSweetArray();
					cin >> choice;
					this->customer[this->size - 1]->RemoveItem(choice-1, cancellationFee);
					priceToPay += cancellationFee;//if it a cookielida so the canncelationfee is 10% now and we have to add it to the pricetopay
				}
			} while (option != Pay);
			system("cls");//clean the screen
			char decision;
			HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
			SetConsoleTextAttribute(hstdout, Purple);
			if (!CheckIfAMember(name))//if he is  a new member,we offer him to join our club
			{
				cout << "You are not a member ,it cost 15 to join.";
				cout << "Would you like to join? Y-yes N-no" << endl;
				cin >> decision;
				if (decision <= 'z' && decision >= 'a')
					decision += 'A' - 'a';
				system("cls");//clean the screen
				if (decision == 'Y')
				{
					priceToPay += MemberPrice;//add 15 to his price to pay
					AddCustomerToMember(name);
					priceToPay = priceToPay - priceToPay * Discount;//offer him 5% off
				}
			}
			else
			{
				priceToPay = priceToPay - priceToPay * Discount;//if he is already a member we give him a 5% off
			}

			cout << "Your bill is: " << fixed << setprecision(1) << priceToPay << ", Thanks you!" << endl;//make the pricetopay with 1 digit after the point
		}
	} while (isANewMember != 'N');
	HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hstdout, RED);
	cout << "Sum of the day:" << endl;
	if (this->size > NoCust)//if there are customers
	{
		cout << "No Customers: " << this->size << endl;
		for (int i = 0; i < this->size; i++)//find the revenue of the day
		{
			for (int j = 0; j < this->customer[i]->GetSize(); j++)
			{
				totalRevenue += this->customer[i]->GetArray()[j]->GetPrice();
			}
		}
		cout << "Total Revenue: " << fixed << setprecision(1) << totalRevenue << ". Total sweets: ";
		this->customer[0]->GetArray()[0]->NoSweet();//print the num of sweets
		cout << endl;
		FindTheBiggestBuyer();
	}
	else
		cout << "No Customers: 0" << endl;//if there are no customers tdy
	cout << "See you tommorow" << endl;
	PrintToFile(totalRevenue, date);//print to sumshop file


}
void SweetShop::PrintToFile(const float& totalrevenue, const String& date)
{
	ofstream myfile("SumShop.txt", ios::app);
	if (myfile.is_open())//open file
	{
		myfile << date <<"Total Revenue: " <<totalrevenue << endl;
		myfile.close();//close file
	}
	else
	{
		cout << "Unable to open file";
	}
}
void SweetShop::FindWhichTypeIsMostBought(const int& index, const float& finalbuy)
{
	float cookieprice = 0;
	float icecreamprice = 0;
	float candyprice = 0;
	float cookielidaprice = 0;
	for (int j = 0; j < this->customer[index]->GetSize(); j++)
	{
		//to check which item is it,if it a cookielida,icecream ,candy or cookie
		bool isCookielida = false;
		Candy * si;
		si = dynamic_cast<Candy*>(this->customer[index]->GetArray()[j]);
		if (si != NULL)
		{
			candyprice += si->GetPrice();
		}
		Cookielida * si3;
		si3 = dynamic_cast<Cookielida*>(this->customer[index]->GetArray()[j]);
		if (si3 != NULL)
		{
			isCookielida = true;
			cookielidaprice += si3->GetPrice();
		}
		if (!isCookielida)//cookielida is also a cookie and icecream ,so to avoid duplicate prints
		{
			Cookie * si1;
			si1 = dynamic_cast<Cookie*>(this->customer[index]->GetArray()[j]);
			if (si1 != NULL)
			{
				cookieprice += si1->GetPrice();
			}
			Icecream * si2;
			si2 = dynamic_cast<Icecream*>(this->customer[index]->GetArray()[j]);
			if (si2 != NULL)
			{
				icecreamprice += si2->GetPrice();
			}
		}
	}
	float temp[TempSize] = { candyprice, cookielidaprice, cookieprice, icecreamprice }; //temp array to find which of the 4  is the biggest
	float max = 0;
	float finalindex = 0;
	for (int i = 0; i < TempSize; i++)
	{
		if (temp[i]>max)
		{
			finalindex = i;
			max = temp[i];
		}
	}
	if (finalindex == IsCandy)//if that candy
	{
		cout <<" Candy= " << fixed << setprecision(1) << (candyprice / finalbuy) * Percentage << '%' << endl;
	}
	else if (finalindex == IsCookielida)//if that cookielida
	{
		cout << " Cookielida= " << fixed << setprecision(1) << (cookielidaprice / finalbuy) * Percentage << '%' << endl;
	}
	else if (finalindex == IsCookie)//if that cookie
	{
		cout << " Cookie= " << fixed << setprecision(1) << (cookieprice / finalbuy) * Percentage << '%' << endl;
	}
	else if (finalindex == IsIcecream)//if that icecream
	{
		cout << " Icecream= " << fixed << setprecision(1) << (icecreamprice / finalbuy) * Percentage << '%' << endl;
	}
}
void SweetShop::FindTheBiggestBuyer()
{
	int finalindex = 0, buyindex = 0;
	float currbuy, finalbuy = 0;
	for (int i = 0; i < this->size; i++)//find the biggest buyer index and amount
	{
		currbuy = 0;
		for (int j = 0; j < this->customer[i]->GetSize(); j++)
		{
			currbuy += this->customer[i]->GetArray()[j]->GetPrice();
		}
		if (CheckIfAMember(this->customer[i]->GetName()))//if he is a member already, he have to get 5% discount,and we have to print it
			currbuy -= currbuy*Discount;
		if (currbuy > finalbuy)
		{
			finalindex = i;
			finalbuy = currbuy;
		}
	}
	float biggestbuy = 0;//find the biggest item in his cart that he paid for it the most
	for (int j = 0; j < this->customer[finalindex]->GetSize(); j++)
	{
		currbuy = this->customer[finalindex]->GetArray()[j]->GetPrice();
		if (biggestbuy < currbuy)
		{
			biggestbuy = currbuy;
			buyindex = j;
		}
	}
	
	cout << "Best Client: " << this->customer[finalindex]->GetName() << "Client no " << this->customer[finalindex]->GetID() << " , Total= " << fixed << setprecision(1) << finalbuy;
	FindWhichTypeIsMostBought(finalindex, finalbuy);
	
}
void SweetShop::AddSweetToArray(const int& index, const Candies& candies)
{
	float amount;
	cout << "Choose how much gram candies you want" << endl;
	cin >> amount;
	Sweetitem * candy = new Candy(candies, amount, CandyPrice);
	if (*(this->customer[index]) != candy)//check if the candy is already in the cart
		*(this->customer[index]) += candy;//if not it will add it
}
void SweetShop::AddSweetToArray(const int& index, const Cookies& cookies)
{
	int amount;
	cout << "Choose how much cookies you want" << endl;
	cin >> amount;
	Sweetitem * cookie = new Cookie(cookies, amount, CookiePrice);
	if (*(this->customer[index]) != cookie)//check if the cookie is already in the cart
		*(this->customer[index]) += cookie;//if not it will add it
}
void SweetShop::AddSweetToArray(const int& index, const Taste& icecreams)
{
	int amount;
	cout << "Choose how much balls you want" << endl;
	cin >> amount;
	Sweetitem * icecream = new Icecream(icecreams, amount, IcecreamPrice);
	if (*(this->customer[index]) != icecream)//check if the icecream is already in the cart
		*(this->customer[index]) += icecream;//if not it will add it
}
void SweetShop::AddSweetToArray(const int& index, const Taste& icecreams, const Cookies& cookies)
{
	Sweetitem * cookielida = new Cookielida(icecreams, cookies);
	if (*(this->customer[index]) != cookielida)//check if the cookielida is already in the cart
		*(this->customer[index]) += cookielida;//if not it will add it
}
void SweetShop::OrderMenu(const int& index)
{
	int sweetoption;
	char anotherSweet = 'Y';
	while (anotherSweet=='Y')
	{
		HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(hstdout, Yellow);
		cout << "Choose which type of sweet you want:" << endl;
		cout << "1-Candy" << endl << "2-Cookie" << endl << "3-Icecream" << endl << "4-Cookielida" << endl;
		cin >> sweetoption;
		int taste;
		//those menus and ifs will ask which sweetitem he wants and which taste
		if (sweetoption == Candiess)
		{
			HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
			SetConsoleTextAttribute(hstdout, Gray);
			cout << "Choose which taste you want:" << endl;
				cout << "1-Chocolate " << CandyPrice << endl << "2-Lolipop " << CandyPrice << endl;
			cout << "3-Snake " << CandyPrice << endl << "4-Teddybear " << CandyPrice << endl;
			cin >> taste;
			if (taste == Candies::Chockolate)
			{
				AddSweetToArray(index, Chockolate);
			}
			else if (taste == Candies::Lolipop)
			{
				AddSweetToArray(index, Lolipop);
			}
			else if (taste == Candies::Snake)
			{
				AddSweetToArray(index, Snake);
			}
			else if (taste == Candies::Teddybear)
			{
				AddSweetToArray(index, Teddybear);
			}
		}
		else if (sweetoption == Cookiess)
		{
			HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
			SetConsoleTextAttribute(hstdout, Blue);
			cout << "Choose which taste you want:" << endl;
			cout << "1-Chocolate " << CookiePrice << endl << "2-Chocolatechips " << CookiePrice << endl << "3-Vanilla " << CookiePrice << endl << "4-Coffee " << CookiePrice << endl;
			cin >> taste;
			if (taste == Cookies::Chocolate)
			{
				AddSweetToArray(index, Chocolate);
			}
			else if (taste == Cookies::Chocolatechips)
			{
				AddSweetToArray(index, Chocolatechips);
			}
			else if (taste == Cookies::Vanilla)
			{
				AddSweetToArray(index, Vanilla);
			}
			else if (taste == Cookies::Coffee)
			{
				AddSweetToArray(index, Coffee);
			}
		}
		else if (sweetoption == Icecreamss)
		{
			HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
			SetConsoleTextAttribute(hstdout, WHITE);
			cout << "Choose which taste you want:" << endl;
			cout << "1-ItalianChocolate " << IcecreamPrice << endl << "2-BulgerianChocolatechips " << IcecreamPrice << endl << "3-GermanVanilla " << IcecreamPrice << endl << "4-RomanianCoffee " << IcecreamPrice << endl;
			cin >> taste;
			if (taste == Taste::ItalianChocolate)
			{
				AddSweetToArray(index, ItalianChocolate);
			}
			else if (taste == Taste::BulgerianChocolatechips)
			{
				AddSweetToArray(index, BulgerianChocolatechips);
			}
			else if (taste == Taste::GermanVanilla)
			{
				AddSweetToArray(index, GermanVanilla);
			}
			else if (taste == Taste::RomanianCoffee)
			{
				AddSweetToArray(index, RomanianCoffee);
			}
		}
		else if (sweetoption == Cookielidass)
		{
			HANDLE hstdout = GetStdHandle(STD_OUTPUT_HANDLE);
			SetConsoleTextAttribute(hstdout, Aqua);
			cout << "Choose which two cookies' taste for cookielida you want"<< endl <<"Pay attention-The Price is for two cookies" << endl;
			cout << "1-Chocolate " << CookiePrice * 2 << endl << "2-Chocolatechips " << CookiePrice * 2 << endl << "3-Vanilla " << CookiePrice * 2 << endl << "4-Coffee " << CookiePrice * 2 << endl;
			cin >> taste;
			if (taste == Cookies::Chocolate)
			{
				cout << "Choose which icecream for cookielida you want:" << endl;
				cout << "1-ItalianChocolate " << IcecreamPrice << endl << "2-BulgerianChocolatechips " << IcecreamPrice << endl << "3-GermanVanilla " << IcecreamPrice << endl << "4-RomanianCoffee " << IcecreamPrice << endl;
				cin >> taste;
				if (taste == Taste::ItalianChocolate)
				{
					AddSweetToArray(index, ItalianChocolate, Chocolate);
				}
				else if (taste == Taste::BulgerianChocolatechips)
				{
					AddSweetToArray(index, BulgerianChocolatechips, Chocolate);
				}
				else if (taste == Taste::GermanVanilla)
				{
					AddSweetToArray(index, GermanVanilla, Chocolate);
				}
				else if (taste == Taste::RomanianCoffee)
				{
					AddSweetToArray(index, RomanianCoffee, Chocolate);
				}
			}
			else if (taste == Cookies::Chocolatechips)
			{
				cout << "Choose which icecream you want:" << endl;
				cout << "1-ItalianChocolate " << IcecreamPrice << endl << "2-BulgerianChocolatechips " << IcecreamPrice << endl << "3-GermanVanilla " << IcecreamPrice << endl << "4-RomanianCoffee " << IcecreamPrice << endl;
				cin >> taste;
				if (taste == Taste::ItalianChocolate)
				{
					AddSweetToArray(index, ItalianChocolate, Chocolatechips);
				}
				else if (taste == Taste::BulgerianChocolatechips)
				{
					AddSweetToArray(index, BulgerianChocolatechips, Chocolatechips);
				}
				else if (taste == Taste::GermanVanilla)
				{
					AddSweetToArray(index, GermanVanilla, Chocolatechips);
				}
				else if (taste == Taste::RomanianCoffee)
				{
					AddSweetToArray(index, RomanianCoffee, Chocolatechips);
				}
			}
			else if (taste == Cookies::Vanilla)
			{
				cout << "Choose which icecream you want:" << endl;
				cout << "1-ItalianChocolate " << IcecreamPrice << endl << "2-BulgerianChocolatechips " << IcecreamPrice << endl << "3-GermanVanilla " << IcecreamPrice << endl << "4-RomanianCoffee " << IcecreamPrice << endl;
				cin >> taste;
				if (taste == Taste::ItalianChocolate)
				{
					AddSweetToArray(index, ItalianChocolate, Vanilla);
				}
				else if (taste == Taste::BulgerianChocolatechips)
				{
					AddSweetToArray(index, BulgerianChocolatechips, Vanilla);
				}
				else if (taste == Taste::GermanVanilla)
				{
					AddSweetToArray(index, GermanVanilla, Vanilla);
				}
				else if (taste == Taste::RomanianCoffee)
				{
					AddSweetToArray(index, RomanianCoffee, Vanilla);
				}
			}
			else if (taste == Cookies::Coffee)
			{
				cout << "Choose which icecream you want:" << endl;
				cout << "1-ItalianChocolate " << IcecreamPrice << endl << "2-BulgerianChocolatechips " << IcecreamPrice << endl << "3-GermanVanilla " << IcecreamPrice << endl << "4-RomanianCoffee " << IcecreamPrice << endl;
				cin >> taste;
				if (taste == Taste::ItalianChocolate)
				{
					AddSweetToArray(index, ItalianChocolate, Coffee);
				}
				else if (taste == Taste::BulgerianChocolatechips)
				{
					AddSweetToArray(index, BulgerianChocolatechips, Coffee);
				}
				else if (taste == Taste::GermanVanilla)
				{
					AddSweetToArray(index, GermanVanilla, Coffee);
				}
				else if (taste == Taste::RomanianCoffee)
				{
					AddSweetToArray(index, RomanianCoffee, Coffee);
				}
			}
		}
		SetConsoleTextAttribute(hstdout, Yellow);
		cout << "Do you want to buy another sweet? Click Y-yes , N-no" << endl;
		cin >> anotherSweet;
		if (anotherSweet <= 'z' && anotherSweet >= 'a')//if he want another item
			anotherSweet += 'A' - 'a';
	}
}
void SweetShop::AddCustomerToMember(const String& name)
{
	ofstream myfile("Subscription.txt", ios::app);
	if (myfile.is_open())//open file
	{
		myfile << name;
		myfile.close();//close file
	}
	else
	{
		cout << "Unable to open file";
	}
}
bool SweetShop::CheckIfAMember(const String& name)
{
	ifstream myfile("Subscription.txt", ios::in);
	if (myfile.is_open())//open file
	{
		char delimeter = '0';
		while (!myfile.eof())
		{
			String check;
			myfile >> check;
			if (name == check)
			{
				myfile.close();//close file
				return true;
			}
			myfile >> delimeter;
		}
		myfile.close();//close file
	}
	else
	{
		cout << "Unable to open file";
	}
	return false;
}
Customer * SweetShop::operator [] (const int& Index) const
{
	assert(Index >= NotPossibleINDEX && Index<this->size);
	return this->customer[Index];//return the customer*  in this index
}
SweetShop::~SweetShop()
{
	if (this->customer)//if the customer isn't empty
	{
		for (int i = 0; i < this->size; i++)//firstly  delete customer*
		{
			delete this->customer[i];
		}
		delete[] this->customer;//then delete customer**
	}
}