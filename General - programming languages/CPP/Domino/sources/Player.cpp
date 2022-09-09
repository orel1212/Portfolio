
#include "Player.h"
#define Size 1
Player::Player() :pile(Size)
{
	this->name = "Computer";//insert into the computer's name , "Computer"
}
Player::Player(char * name) : pile(Size)
{
	this->name = name;//insert into the player name his name
}