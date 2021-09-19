/* Assignment: 2
Campus: Beer-Sheva
Author1: Orel Lavie, ID: 207632118
Date:7/4 /2015 */
#include "Game.h"
int main()
{
	char name[80];
	cout << "Welcome to a domino game!" << endl;
	cout << "enter your name" << endl;
	cin >> name;
	Game game(name);
	game.run();
}