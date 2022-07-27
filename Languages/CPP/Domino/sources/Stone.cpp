#include "Stone.h"
Stone::Stone()
{}
Stone::Stone(Stone& obj)
{
	this->left = obj.left;
	this->right = obj.right;
}
void Stone::Reversal()
{
	int temp;
	temp = this->left;
	this->left = this->right;
	this->right = temp;
}
bool Stone::Compare(Stone& obj)
{
	if (this->left == obj.left && this->right == obj.right)
		return true;
	else
		return false;
}