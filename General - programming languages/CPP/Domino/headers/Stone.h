#ifndef STONE_H
#define STONE_H
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
typedef enum {ZERO,ONE,TWO,THREE,FOUR,FIVE,SIX } OneToSix;
using namespace std;
class Stone
{
	int right;
	int left;
public:
	Stone();//default constructor that doing nothing and get nothing
	Stone(Stone& obj);//copy constructor that get reference to obj and create new stone with the obj's values
	void Reversal();//reverse the stone from right to left and from left to right
	inline void SetStone(int left, int right)//set new values to stone, get the new left and right
	{
		this->left = left;
		this->right = right;
	}
	inline void StoneOpenPrint()//print the stone openned with values inside
	{
		cout << "[" << this->left << "][" << this->right << "]";
	}
	inline void StoneClosePrint()//print the stone closed without values,they're hidded
	{
		cout << "[][]";
	}

	bool Compare(Stone& obj);//compare between this object and the object that the function get,return true if they are similiar
	inline int GetRight()//return the value of the right of the stone
	{
		return this->right;
	}
	inline int GetLeft()//return the value of the left of the stone
	{
		return this->left;
	}
};
#endif