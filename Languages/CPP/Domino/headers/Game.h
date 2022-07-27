#ifndef GAME_H
#define GAME_H
#include "Player.h"
class Game
{
	Player player;
	Player computer;
	Pile pile;
	Pile gamePile;
	int right;
	int left;
	bool GameOver();//return true if the game is over and have to announce the winner
	void PrintGame();//print the pile of the player,the computer and the game pile
public:
	Game();//c'tor default,doing nothing
	Game(char * name);//c'tor that get the name of the player, and create the player,the computer, the gamepile and the pile
	void run();//function that manage the game
	
};
#endif;