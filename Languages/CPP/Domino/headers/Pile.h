#ifndef PILE_H
#define PILE_H
#include "Stone.h"
#include <time.h>
#define Loop 7
typedef enum { Left,Right } LeftOrRight;
class Pile
{
	Stone * pile;
	int size;//size of the stone array
public:
	Pile();//default constructor,doing nothing
	Pile(int size);//constructor that get size of array and make new array of stones
	Stone& GetStone(int index);//return reference to the stone in the array,get index of the place of the stone in the array
	inline int GetSize()//return the size of the pile
	{
		return this->size;
	}
	void PileOpenPrint();//print the array openned, uses the pileopenprint() function of stone class
	void PileClosePrint();//print the array closed, uses the pilecloseprint() function of stone class
	void InsertToPile();//make new pile of 28 stones
	void Mix();//mix the stones
	void AddStoneToArray(Stone& obj, LeftOrRight index);//get new stone reference,and index if right or left , cuz it depends where to add it,for example 1-1 and 1-3 is right(1-3) but 1-3 and 1-1 is left(1-1)
	void RemoveStoneFromTheArray(Stone& obj);//remove the stone from the array that similiar to obj using compare function
	bool Compare(Stone& obj, int index);//compare function that get index and obj reference, uses the compare function in stone class, return true if they're similiar
	~Pile();//d'tor
	
};
#endif