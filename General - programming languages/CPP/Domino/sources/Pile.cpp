#include "Pile.h"
Pile::Pile(int size):size(size)//using intilize line to enter the size to this->size
{
	this->pile = new  Stone[this->size];
	if (this->size == 1)
		this->size = 0;//the size of the array is 1, but it still filled with garbitch, which can affect the AddStoneToArray function when we move from 1fields to 2fields.
}
Pile::Pile()
{
}
Stone& Pile::GetStone(int index)
{
	if (index >= size)//if the index is biggest from size-1(the highest index in the array) so there is problem there and we must close the program
	{
		cout << "There is no match to this index" << endl;
		exit(1);
	}
	for (int i = 0; i < this->size; i++)
	{
		if (i == index)//if the index is this i, so we return the stone in this index
		{
			return this->pile[i];
		}
	}
}
void Pile::PileOpenPrint()
{
	for (int i = 0; i < this->size; i++)
	{
		cout << i + 1 << ".";//print the index to the stone : 1 2 3 4 5 ....
		this->pile[i].StoneOpenPrint();
		if ((i + 1) % 7 == 0)//after 7 stones there is a endl
			cout << endl;
		else
			cout << " ";//between each stone there is space
	}
}
void Pile::PileClosePrint()
{
	for (int i = 0; i < this->size; i++)
	{
		cout << i + 1 << ".";//print the index to the stone : 1 2 3 4 5 ....
		this->pile[i].StoneClosePrint();
		if ((i + 1) % 7 == 0)//after 7 stones there is a endl
			cout << endl;
		else
			cout << " ";//between each stone there is space
	}
}
void Pile::InsertToPile()
{
	int index = 0;
	for (int i = 0; i < Loop; i++)//from 0 to 6 each stone ,0-0 to 6-6
	{
		for (int j = 0; j <= i; j++)
		{
			this->pile[index].SetStone(i,j);
			index++;
		}
	}
}
void Pile::Mix()
{
	int index;
	/* initialize random seed: */
	srand(time(NULL));
	for (int i = 0; i < this->size; i++)
	{
		/* generate random number between 0 and 27: */
		index = rand() % 27;
		Stone temp=this->pile[i];//we make a temp and swap with the two we wanted using the index of the random
		this->pile[i] = Stone(this->pile[index]);
		this->pile[index] = Stone(temp);
	}
}
void Pile::AddStoneToArray(Stone& obj, LeftOrRight index)
{
	if (index == Right)//if we want to add the stone to the right
	{
		if (this->size == 0)//if there is just 1 stone so we just using the copy c'tor and add 1 to the array
		{
			this->pile[0] = Stone(obj);
			this->size += 1;
		}
		else if (this->size > 0)//if there are more than 1 stone
		{
			Stone * stonecpy = new Stone[this->size + 1];// we create temp array that biggest from the array by 1
			for (int i = 0; i < this->size; i++)//using c'tor to copy from one array to another
			{
				stonecpy[i] = Stone(this->pile[i]);
			}
			delete[] this->pile;//free the memory we don't need
			stonecpy[this->size] = Stone(obj);//using c'tor to add the obj to the last field in the temp array
			this->pile = new Stone[this->size + 1];//creating new main array with the size+1
			for (int i = 0; i < this->size + 1; i++)//using copy c'tor to copy from the temp to the main
				this->pile[i] = Stone(stonecpy[i]);
			delete[]stonecpy;//delete the temp memory
			this->size += 1;
		}
	}
	else if (index == Left)//if we want to add the stone to the left
	{
		Stone * stonecpy = new Stone[this->size + 1];// we create temp array that biggest from the array by 1
		for (int i = 0; i < this->size; i++)
		{
			stonecpy[i+1] = Stone(this->pile[i]);//using c'tor to copy from one array to another, we start the temp array from the second field, cuz the first will be filled with the new obj
		}
		delete[] this->pile;
		stonecpy[0] = Stone(obj);//using c'tor to add the obj to the first field in the temp array
		this->pile = new Stone[this->size + 1];//creating new main array with the size+1
		for (int i = 0; i < this->size + 1; i++)//using copy c'tor to copy from the temp to the main
			this->pile[i] = Stone(stonecpy[i]);
		delete[]stonecpy;//delete the temp memory
		this->size += 1;
	}
	
}
void Pile::RemoveStoneFromTheArray(Stone& obj)
{
	if (this -> size > 1)//if there are more than 1 fields in the array
	{
		Stone * stonecpy = new Stone[this->size - 1];//create temp array to arrive those stones without the one we want to delete
		for (int i = 0,j=0; i < this->size; i++)
		{
			if (this->pile[i].Compare(obj) == false)//if the stone in the array isn't similiar to the obj we copy it
			{
				stonecpy[j] = Stone(this->pile[i]);
				j++;
			}
		}
		delete[] this->pile;//delete the main memory
		this->pile = new Stone[this->size - 1];//create new one with 1less field
		for (int i = 0; i < this->size - 1; i++)//using copy c'tor to copy from the temp array to the main array the stones
			this->pile[i] = Stone(stonecpy[i]);
		delete[] stonecpy;//free the temp
		this->size -= 1;
	}
	else if(this->size==1)//there is no need to delete it if there is only 1 stone, it will be deleted by the de'tor
	{
		if (this->pile[0].Compare(obj) == true)
		{
			this->size -= 1;
		}
	}
}
bool Pile::Compare(Stone& obj, int index)
{
	return (this->pile[index].Compare(obj));//return true if they are similiar
}
Pile::~Pile()
{
	delete[] this->pile;
}