
#ifndef PLAYER_H
#define PLAYER_H
#include "Pile.h"
#include<string>
using namespace std;
class Player
{
	string name;
	Pile pile;
public:
	Player();//using pile c'tor to create the computer
	Player(char * name);//using pile c'tor to create the player,get his name
	inline void AddStoneToArray(Stone& obj, LeftOrRight index)//get obj reference and index, using addstonetoarray function in pile
	{
		this->pile.AddStoneToArray(obj,index);
	}
	inline void RemoveStoneFromTheArray(Stone& obj)//get obj reference, using RemoveStoneFromTheArray function in pile
	{
		this->pile.RemoveStoneFromTheArray(obj);
	}
	inline void PileOpenPrint()//print the name of the player and his pile using fileopenprint from pile class
	{
		cout << "---" << this->name << "---" << endl;
		this->pile.PileOpenPrint();
	}
	inline void PileClosePrint()//print the computer and his pile using filecloseprint from pile class
	{
		cout << "---Computer---" << endl;
		this->pile.PileClosePrint();
	}
	inline bool Compare(Stone& obj,int index)//get obj reference and index, using compare function in pile ,return true if they are similiar
	{
		return (this->pile.Compare(obj,index));
	}
	inline Stone& GetStone(int index)//return the stone reference using index , that function using the getstone function in pile class
	{
		return this->pile.GetStone(index);
	}
	inline int GetSize()//return the size of the player's array, using getsize function in class pile
	{
		return this->pile.GetSize();
	}
	
	
};
#endif