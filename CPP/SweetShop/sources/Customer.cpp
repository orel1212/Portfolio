
#include"Customer.h"
Customer::Customer(const String& name, const int& id)
{
	this->name = name;
	this->id = id;
	this->array = new Sweetitem *[DefaultSize];//create array with size 1
	this->size = NoCust;//still no customers
}
Sweetitem * Customer::operator [] (const int& Index) const
{
	assert(Index >= NotPossibleINDEX && Index<this->size);
	return this->array[Index];//return the sweetitem*  in this index
}
Customer& Customer::operator+=(Sweetitem * SI)
{
	if (this->size == NoCust)//if there are no customer so no items
	{
		this->array[0] = SI;
		this->size += 1;
		this->array[0]->AddOneToSNUM();
	}
	else//copy the sweet items of the customer before adding new sweetitem
	{
			this->array[0]->AddOneToSNUM();
			Sweetitem ** temp = new Sweetitem *[this->size + 1];
			for (int i = 0; i < this->size; i++)
			{
				temp[i] = this->array[i];
			}
			temp[this->size] = SI;
			delete[] this->array;
			this->size += 1;
			this->array = new Sweetitem *[this->size];
			for (int i = 0; i < this->size; i++)
			{
				this->array[i] = temp[i];
			}
			delete[] temp;
	}
	return *this;
}
void Customer::RemoveItem(const int& Index,float& CancellationFee)
{
	assert(Index >= NotPossibleINDEX && Index<this->size);
		Sweetitem ** temp = new Sweetitem *[this->size - 1];
		for (int i = 0,j=0; i < this->size; i++)//copy the sweet items without the one we want to remove
		{
			if (i != Index)
			{
				temp[j] = this->array[i];
				j++;
			}
			else//check before we remove the sweetitem if it a cookielida
			{
				Cookielida * si;
				si = dynamic_cast<Cookielida*>(this->array[i]);
				if (si != NULL)
				{
					CancellationFee = si->CancellationFee();//the customer have to pay cancellation fee if it a cookielida
					cout << "Cancellation Fee is :" << CancellationFee << endl;
				}
				delete this->array[i];
			}
		}
		delete[] this->array;
		this->size -= 1;
		this->array = new Sweetitem *[this->size];
		for (int i = 0; i < this->size; i++)
		{
			this->array[i] = temp[i];
		}
		delete[] temp;
		this->array[0]->ReduceOneFromSNUM();//reduce one sweetitem cuz we deleted one sweetitem
}
void Customer::PrintSweetArray()
{
	bool checkifcookielida;
	for (int i = 0; i < this->size; i++)
	{
		checkifcookielida = false;//check if it cookielida so it won't check if it a cookie and icecream,cuz if it a cookielida so it also a cookie and a icecream
		Cookielida * s4;
		s4 = dynamic_cast<Cookielida*>(this->array[i]);
		if (s4 != NULL)//rtti
		{
			checkifcookielida = true;
			if (s4->GetCookielidaIcecream() == Taste::ItalianChocolate)
			{
				if (s4->GetCookielidaCookies() == Cookies::Chocolate)
					cout << i + 1 << ".Cookielida: 2 Chocolate cookies and 1 ItalianChocolate icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Chocolatechips)
					cout << i + 1 << ".Cookielida: 2 Chocolatechips cookies and 1 ItalianChocolate icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Coffee)
					cout << i + 1 << ".Cookielida: 2 Coffee cookies and 1 ItalianChocolate icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Vanilla)
					cout << i + 1 << ".Cookielida: 2 Vanilla cookies and 1 ItalianChocolate icecream" << endl;
			}
			if (s4->GetCookielidaIcecream() == Taste::BulgerianChocolatechips)
			{
				if (s4->GetCookielidaCookies() == Cookies::Chocolate)
					cout << i + 1 << ".Cookielida: 2 Chocolate cookies and 1 BulgerianChocolatechips icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Chocolatechips)
					cout << i + 1 << ".Cookielida: 2 Chocolatechips cookies and 1 BulgerianChocolatechips icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Coffee)
					cout << i + 1 << ".Cookielida: 2 Coffee cookies and 1 BulgerianChocolatechips icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Vanilla)
					cout << i + 1 << ".Cookielida: 2 Vanilla cookies and 1 BulgerianChocolatechips icecream" << endl;
			}
			if (s4->GetCookielidaIcecream() == Taste::RomanianCoffee)
			{
				if (s4->GetCookielidaCookies() == Cookies::Chocolate)
					cout << i + 1 << ".Cookielida: 2 Chocolate cookies and 1 RomanianCoffee icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Chocolatechips)
					cout << i + 1 << ".Cookielida: 2 Chocolatechips cookies and 1 RomanianCoffee icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Coffee)
					cout << i + 1 << ".Cookielida: 2 Coffee cookies and 1 RomanianCoffee icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Vanilla)
					cout << i + 1 << ".Cookielida: 2 Vanilla cookies and 1 RomanianCoffee icecream" << endl;
			}
			if (s4->GetCookielidaIcecream() == Taste::GermanVanilla)
			{
				if (s4->GetCookielidaCookies() == Cookies::Chocolate)
					cout << i + 1 << ".Cookielida: 2 Chocolate cookies and 1 GermanVanilla icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Chocolatechips)
					cout << i + 1 << ".Cookielida: 2 Chocolatechips cookies and 1 GermanVanilla icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Coffee)
					cout << i + 1 << ".Cookielida: 2 Coffee cookies and 1 GermanVanilla icecream" << endl;
				if (s4->GetCookielidaCookies() == Cookies::Vanilla)
					cout << i + 1 << ".Cookielida: 2 Vanilla cookies and 1 GermanVanilla icecream" << endl;
			}
		}
		Candy * si;
		si = dynamic_cast<Candy*>(this->array[i]);
		if (si != NULL)//rtti
		{
			if (si->GetCandy() == Chockolate)
			{
				cout << i + 1 << ".Candy : Chocolate," << si->GetWeight() << " grams" << endl;
			}
			if (si->GetCandy() == Lolipop)
			{
				cout << i + 1 << ".Candy : Lolipop," << si->GetWeight() << " grams" << endl;
			}
			if (si->GetCandy() == Teddybear)
			{
				cout << i + 1 << ".Candy : Teddybear," << si->GetWeight() << " grams" << endl;
			}
			if (si->GetCandy() == Snake)
			{
				cout << i + 1 << ".Candy : Snake," << si->GetWeight() << " grams" << endl;
			}
		}
		if (!checkifcookielida)
		{
			Cookie * si2;
			si2 = dynamic_cast<Cookie*>(this->array[i]);
			if (si2 != NULL)
			{
				if (si2->GetCookie() == Cookies::Chocolate)
				{
					cout << i + 1 << ".Cookie : Chocolate," << si2->GetNumOfCookies() << " cookies" << endl;
				}
				if (si2->GetCookie() == Cookies::Chocolatechips)
				{
					cout << i + 1 << ".Cookie : Chocolatechips," << si2->GetNumOfCookies() << " cookies" << endl;
				}
				if (si2->GetCookie() == Cookies::Coffee)
				{
					cout << i + 1 << ".Cookie : Coffee," << si2->GetNumOfCookies() << " cookies" << endl;
				}
				if (si2->GetCookie() == Cookies::Vanilla)
				{
					cout << i + 1 << ".Cookie : Vanilla," << si2->GetNumOfCookies() << " cookies" << endl;
				}
			}
			Icecream * si3;
			si3 = dynamic_cast<Icecream*>(this->array[i]);
			if (si3 != NULL)
			{
				if (si3->GetTaste() == Taste::ItalianChocolate)
				{
					cout << i + 1 << ".Icecream : ItalianChocolate," << si3->GetNumOfBalls() << " icecream balls" << endl;
				}
				if (si3->GetTaste() == Taste::BulgerianChocolatechips)
				{
					cout << i + 1 << ".Icecream : BulgerianChocolatechips," << si3->GetNumOfBalls() << " icecream balls" << endl;
				}
				if (si3->GetTaste() == Taste::RomanianCoffee)
				{
					cout << i + 1 << ".Icecream : RomanianCoffee," << si3->GetNumOfBalls() << " icecream balls" << endl;
				}
				if (si3->GetTaste() == Taste::GermanVanilla)
				{
					cout << i + 1 << ".Icecream : GermanVanilla," << si3->GetNumOfBalls() << " icecream balls" << endl;
				}
			}
		}
	}
}
bool Customer::operator!=(Sweetitem * check)
{
	bool foundarray;
	if (this->size == NoCust)
		return true;
	else
	{
		for (int i = 0; i < this->size; i++)
		{
			foundarray = false;
			Candy * help;
			help = dynamic_cast<Candy*>(this->array[i]);
			if (help != NULL)
			{
				foundarray = true;
				Candy * si;
				si = dynamic_cast<Candy*>(check);
				if (si != NULL)
				{
					if (help->GetCandy() == si->GetCandy() && help->GetWeight() == si->GetWeight())//check if they're similiar
					{
						return false;
						break;
					}
				}
			}
			if (!foundarray)//if it not candy
			{
				Cookielida * help;
				help = dynamic_cast<Cookielida*>(this->array[i]);
				if (help != NULL)
				{
					foundarray = true;
					Cookielida * si;
					si = dynamic_cast<Cookielida*>(check);
					if (si != NULL)
					{
						if (help->GetCookielidaIcecream() == si->GetCookielidaIcecream() && help->GetCookielidaCookies() == si->GetCookielidaCookies())//check if they're similiar
						{
							return false;
							break;
						}
					}
				}
			}
			if (!foundarray)//if it not cookielida
			{
				Cookie * help;
				help = dynamic_cast<Cookie*>(this->array[i]);
				if (help != NULL)
				{
					foundarray = true;
					Cookie * si;
					si = dynamic_cast<Cookie*>(check);
					if (si != NULL)
					{
						if (help->GetCookie() == si->GetCookie() && help->GetNumOfCookies() == si->GetNumOfCookies())//check if they're similiar
						{
							return false;
							break;
						}
					}
				}
			}
			if (!foundarray)//if it not cookie
			{
				Icecream * help;
				help = dynamic_cast<Icecream*>(this->array[i]);
				if (help != NULL)
				{
					foundarray = true;
					Icecream * si;
					si = dynamic_cast<Icecream*>(check);
					if (si != NULL)
					{
						if (help->GetTaste() == si->GetTaste() && help->GetNumOfBalls() == si->GetNumOfBalls())//check if they're similiar
						{
							return false;
							break;
						}
					}
				}
			}
		}
		return true;
	}
}
Customer::~Customer()
{
	for (int i = 0; i < this->size; i++)//delete firstly the sweetitem*
	{
		delete this->array[i];
	}
	delete[] this->array;//delete sweetitem **
}