/* Assignment: 2
Campus: Beer-Sheva
Author1: Orel Lavie, ID: 207632118
Date:7/4 /2015 */
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