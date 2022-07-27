
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