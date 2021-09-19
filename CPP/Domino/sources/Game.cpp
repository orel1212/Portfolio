
#include "Game.h"
#define PSize 28
#define Start 1
#define StartCopy 27
#define SecondCopy 20
#define Stop 13
Game::Game()
{}
Game::Game(char * name) :player(name), computer(), pile(PSize), gamePile(Start)//create player,computer,pile,gamepile
{
}
void Game::PrintGame()
{
	cout << " KUPA" << endl;
	this->pile.PileClosePrint();
	cout << endl << endl;
	this->computer.PileClosePrint();
	cout << endl << endl;
	cout << "---Game Pile---" << endl;
	this->gamePile.PileOpenPrint();
	cout << endl << endl;
	this->player.PileOpenPrint();
	cout << endl << endl;
}
bool Game::GameOver()
{
	if (this->player.GetSize() == 0)//if the player pile is empty
	{
		int sum = 0;
		for (int i = 0; i < this->computer.GetSize(); i++)
		{
			sum += this->computer.GetStone(i).GetLeft() + this->computer.GetStone(i).GetRight();//calc his score
		}
		system("cls");
		cout << "Congrats , You have won the computer , your score is : " << sum << endl;
		return true;
	}
	else if (this->computer.GetSize() == 0)//if the computer pile is empty
	{
		int sum = 0;
		for (int i = 0; i < this->player.GetSize(); i++)
		{
			sum += this->player.GetStone(i).GetLeft() + this->player.GetStone(i).GetRight();//calc computer score
		}
		system("cls");
		cout << "Sorry, the computer won , the score of the computer is : " << sum << endl;
		return true;
	}
	else if (this->pile.GetSize() == 0)//if the main pile is empty
	{
		bool checkc = true;
		bool checkp = true;
		for (int i = 0; i < player.GetSize(); i++)//check the player pile if there is somthing more to put on the gamepile
		{
			if (this->player.GetStone(i).GetLeft() == this->right)
			{
				checkp = false;
				break;
			}
			else if (this->player.GetStone(i).GetRight() == this->left)
			{
				checkp = false;
				break;
			}
			else
			{
				this->player.GetStone(i).Reversal();
				if (this->player.GetStone(i).GetLeft() == this->right)
				{
					this->player.GetStone(i).Reversal();
					checkp = false;
					break;
				}
				else if (this->player.GetStone(i).GetRight() == this->left)
				{
					this->player.GetStone(i).Reversal();
					checkp = false;
					break;
				}
			}
		}
		for (int i = 0; i < computer.GetSize(); i++)//check the computer pile if there is somthing more to put on the gamepile
		{
			if (this->computer.GetStone(i).GetLeft() == this->right)
			{
				checkc = false;
				break;
			}
			else if (this->computer.GetStone(i).GetRight() == this->left)
			{
				checkc = false;
				break;
			}
			else
			{
				this->computer.GetStone(i).Reversal();
				if (this->computer.GetStone(i).GetLeft() == this->right)
				{
					this->computer.GetStone(i).Reversal();
					checkc = false;
					break;
				}
				else if (this->computer.GetStone(i).GetRight() == this->left)
				{
					this->computer.GetStone(i).Reversal();
					checkc = false;
					break;
				}
			}
		}
		if (!checkc && !checkp)//if there is more to put so the game isn't end yet
			return false;
		else//the game is ended and the player who has less points he is the winner
		{
			int sump = 0, sumc = 0;
			for (int i = 0; i < this->computer.GetSize(); i++)
			{
				sump += this->computer.GetStone(i).GetLeft() + this->computer.GetStone(i).GetRight();//calc player score
			}
			
			for (int i = 0; i < this->player.GetSize(); i++)
			{
				sumc += this->player.GetStone(i).GetLeft() + this->player.GetStone(i).GetRight();//calc computer score
			}
			if (sump > sumc)//if player score is highest than computer score ,the player is the winner
			{
				system("cls");
				cout << "Congrats , You have won the computer , your score is : " << sump << endl;
			}
			else if (sumc > sump)//if computer score is highest than player score ,the computer is the winner
			{
				system("cls");
				cout << "Sorry, the computer won , the score of the computer is : " << sumc << endl;
			}
			else//there is a draw
			{
				system("cls");
				cout << "This is a draw" << endl;
			}
			return true;

		}
	}
	else
		return false;
}
void Game::run()
{
	this->pile.InsertToPile();
	this->pile.Mix();
	int i = StartCopy;
	while (i > Stop)//take 14 stones from the main pile and give 7 to each player and computer
	{
		if (i > SecondCopy)
		{
			this->player.AddStoneToArray(this->pile.GetStone(i), Right);
			this->pile.RemoveStoneFromTheArray(this->pile.GetStone(i));
		}
		else
		{
			this->computer.AddStoneToArray(this->pile.GetStone(i), Right);
			this->pile.RemoveStoneFromTheArray(this->pile.GetStone(i));
		}
		i--;
	}
	Stone temp;//temp of stone that 6-6 , to compare it with the player and the computer to find who of them had 6-6
	temp.SetStone(6, 6);
	bool cmove = false;//check if the computer played
	bool pmove = false;//check if the player played
	bool ToPrint = false;//check if to print to the player which window,more info is in the player part at the while loop
	bool end = false;//check if the game ends, to prevent many times check if the gameend using Gameover() function
	bool reversec = true;//to check if need to do reverse after we reversed the stone|computer
	bool reversep = true;//to check if need to do reverse after we reversed the stone|player
	system("cls");//clean the screen
	PrintGame();
	for (i = 0; i < Loop; i++)
	{
		if (player.Compare(temp, i))//to compare it with the player  to find if he  had 6 - 6
		{
			cout << "You Start the game, please enter the index of the 6-6 from 1-7" << endl;
			int x;
			cin >> x;
			fflush(stdin);
			this->gamePile.AddStoneToArray(this->player.GetStone(i), Right);
			this->player.RemoveStoneFromTheArray(this->player.GetStone(i));
			this->left = 6;
			this->right = 6;
            pmove = true;
			break;
		}
		else if (computer.Compare(temp, i))//to compare it with the computer  to find if he  had 6 - 6
		{
			this->gamePile.AddStoneToArray(this->computer.GetStone(i), Right);
			this->computer.RemoveStoneFromTheArray(this->computer.GetStone(i));
			this->left = 6;
			this->right = 6;
			cmove = true;
			break;
		}
	}
	if (!pmove && !cmove)//if none of them had 6-6 so the player is starting the game
	{
		cout << "You Start the game, please enter the index of any stone you want" << endl;
		int x;
		cin >> x;
		this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Right);
		this->left = this->player.GetStone(x - 1).GetLeft();
		this->right = this->player.GetStone(x - 1).GetRight();
		this->player.RemoveStoneFromTheArray(this->player.GetStone(x-1));
		pmove = true;
	}
	while (!end)//the game continues until the gameover return true to end
	{
		if (this->computer.GetSize() == 0 || this->player.GetSize() == 0 || this->pile.GetSize() == 0)//if the computer/player's pile or main pile is empty, have to check if finished the game
			end = GameOver();
		if (!cmove && pmove && !end)//if the computer didn't play yet and the game isn't end
		{
			system("cls");
			PrintGame();
			for (int i = 0; i < this->computer.GetSize(); i++)//loop to check which stone in the array, at which state(reversal or not) fit in the gamepile
			{
				if (this->computer.GetStone(i).GetLeft() == this->right)
				{
					this->right = this->computer.GetStone(i).GetRight();
					this->gamePile.AddStoneToArray(this->computer.GetStone(i), Right);
					this->computer.RemoveStoneFromTheArray(this->computer.GetStone(i));
					cmove = true;
					pmove = false;
					break;
				}
				else if (this->computer.GetStone(i).GetRight() == this->left)
				{
					this->left = this->computer.GetStone(i).GetLeft();
					this->gamePile.AddStoneToArray(this->computer.GetStone(i), Left);
					this->computer.RemoveStoneFromTheArray(this->computer.GetStone(i));
					cmove = true;
					pmove = false;
					break;
				}
				else
				{
					this->computer.GetStone(i).Reversal();
					reversec = false;
					if (this->computer.GetStone(i).GetLeft() == this->right)
					{
						this->gamePile.AddStoneToArray(this->computer.GetStone(i), Right);
						this->right = this->computer.GetStone(i).GetRight();
						this->computer.GetStone(i).Reversal();
						reversec = true;
						this->computer.RemoveStoneFromTheArray(this->computer.GetStone(i));
						cmove = true;
						pmove = false;
						break;
					}
					else if (this->computer.GetStone(i).GetRight() == this->left)
					{
						this->gamePile.AddStoneToArray(this->computer.GetStone(i), Left);
						this->left = this->computer.GetStone(i).GetLeft();
						this->computer.GetStone(i).Reversal();
						reversec = true;
						this->computer.RemoveStoneFromTheArray(this->computer.GetStone(i));
						cmove = true;
						pmove = false;
						break;
					}
					if (!reversec)
						this->computer.GetStone(i).Reversal();
				}
			}
			if (this->computer.GetSize() == 0)//if after the computer put the stone , his pile is empty have to check if the game end
			{
				system("cls");
				end = GameOver();
				break;
			}
			else if (!cmove && pmove && this->pile.GetSize() > 0)//if the computer have nothing to put,so he have to take from the mainpile a stone
			{
				this->computer.AddStoneToArray(this->pile.GetStone(this->pile.GetSize() - 1), Right);
				this->pile.RemoveStoneFromTheArray(this->pile.GetStone(this->pile.GetSize() - 1));
			}
			else if (!cmove && pmove)//if the computer didn't play and he have nothing to take from the mainpile,must check if the game is ended 
				end = GameOver();
		}
		if (!pmove && cmove && !end)
		{
			system("cls");
			if (ToPrint)//if the player choosed wrong stone that not fit, he have to decide again
			{
				PrintGame();
				cout << "Choose the domino that fitting in the board" << endl;
				if (this->pile.GetSize() > 0)
					cout << "If you Do not have any domino that fit, click 0 to take domino from the pile" << endl;
				else
					cout << "The stones in the pile are gone" << endl;
			}
			else//if firstly he not choose yet which stone to put, so that have to printout to him on the screen
			{
				PrintGame();
				cout << "Your turn, please enter the index of any stone you want" << endl;
				if (this->pile.GetSize() > 0)
					cout << "If you Do not have any domino that fit, click 0 to take domino from the pile" << endl;
				else
					cout << "The stones in the pile are gone" << endl;
			}
			ToPrint = false;
			if (this->player.GetSize() == 0)//if the player pile is empty must check if the game is ended
			{
				system("cls");
				end = GameOver();
			}
			if (!end)//if the game isn't ended
			{
				int x;
				cin >> x;
				if (this->pile.GetSize() > 0 && x == 0)//if the player have nothing to put,so he have to take from the mainpile a stone
				{
					this->player.AddStoneToArray(this->pile.GetStone(this->pile.GetSize() - 1), Right);
					this->pile.RemoveStoneFromTheArray(this->pile.GetStone(this->pile.GetSize() - 1));
				}
				else if (this->pile.GetSize() == 0 && x == 0)//if the mainpile is empty and he want to take from mainpile a stone, cuz none in his pile is fit, so must check if the game is ended
					end = GameOver();
				else if (this->player.GetStone(x - 1).GetLeft() == this->right && this->player.GetStone(x - 1).GetRight() == this->left)//check if he can put the stone in both sides,like 6 is this->left and 5 is this->right,and his stone is 5-6
				{
					int direction;
					do
					{
						cout << "To put the stone in left - click 0, right- click 1" << endl;
						cin >> direction;
						if (direction == 0)//if he chose left
						{
							this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Left);
							this->left = this->player.GetStone(x - 1).GetLeft();
						}
						else if (direction == 1)//if he chose right
						{
							this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Right);
							this->right = this->player.GetStone(x - 1).GetRight();
						}
					} while (direction != Left && direction != Right);//dowhile snooze loop,if he won't choose 0 or 1
					this->player.RemoveStoneFromTheArray(this->player.GetStone(x - 1));
					pmove = true;
					cmove = false;
				}
				else if (this->player.GetStone(x - 1).GetLeft() == this->left && this->player.GetStone(x - 1).GetRight() == this->right)//check if he can put the stone in both sides,like 6 is this->left and 5 is this->right,and his stone is 6-5
				{
					this->player.GetStone(x - 1).Reversal();
					int direction;
					do
					{
						cout << "To put the stone in left - click 0, right- click 1" << endl;
						cin >> direction;
						if (direction == 0)//if he chose left
						{
							this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Left);
							this->left = this->player.GetStone(x - 1).GetLeft();
						}
						else if (direction == 1)//if he chose right
						{
							this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Right);
							this->right = this->player.GetStone(x - 1).GetRight();
						}
					} while (direction != Left && direction != Right);//dowhile snooze loop,if he won't choose 0 or 1
					this->player.GetStone(x - 1).Reversal();
					this->player.RemoveStoneFromTheArray(this->player.GetStone(x - 1));
					pmove = true;
					cmove = false;
				}
				else if (this->player.GetStone(x - 1).GetRight() == this->right && this->player.GetStone(x - 1).GetRight() == this->left)//if this->right=this->left=6, and he has 6-5
				{
					int direction;
					do
					{
						cout << "To put the stone in left - click 0, right- click 1" << endl;
						cin >> direction;
						if (direction == 0)//if he chose left
						{
							
							this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Left);
							this->left = this->player.GetStone(x - 1).GetLeft();
						}
						else if (direction == 1)//if he chose right
						{
							this->player.GetStone(x - 1).Reversal();//need to reverse if this->right is 6 ,this->left =6, and stone is 5-6
							this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Right);
							this->right = this->player.GetStone(x - 1).GetRight();
							this->player.GetStone(x - 1).Reversal();//reverse back
						}
					} while (direction != Left && direction != Right);//dowhile snooze loop,if he won't choose 0 or 1
					this->player.RemoveStoneFromTheArray(this->player.GetStone(x - 1));
					pmove = true;
					cmove = false;
				}
				else if (this->player.GetStone(x - 1).GetLeft() == this->right && this->player.GetStone(x - 1).GetLeft() == this->left)//if this->right=this->left=6, and he has 6-5
				{
					int direction;
					do
					{
						cout << "To put the stone in left - click 0, right- click 1" << endl;
						cin >> direction;
						if (direction == 0)//if he chose left
						{
							this->player.GetStone(x - 1).Reversal();//need to reverse if this->right is 6 ,this->left =6, and stone is 6-5
							this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Left);
							this->left = this->player.GetStone(x - 1).GetLeft();
							this->player.GetStone(x - 1).Reversal();//reverse back
						}
						else if (direction == 1)//if he chose right
						{
							this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Right);
							this->right = this->player.GetStone(x - 1).GetRight();
						}
					} while (direction != Left && direction != Right);//dowhile snooze loop,if he won't choose 0 or 1
					this->player.RemoveStoneFromTheArray(this->player.GetStone(x - 1));
					pmove = true;
					cmove = false;
				}
				else// to check the stone in the index that the player chose at which state(reversal or not) fit in the gamepile
				{
					if (this->player.GetStone(x - 1).GetLeft() == this->right)
					{
						this->right = this->player.GetStone(x - 1).GetRight();
						this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Right);
						this->player.RemoveStoneFromTheArray(this->player.GetStone(x - 1));
						pmove = true;
						cmove = false;
					}
					else if (this->player.GetStone(x - 1).GetRight() == this->left)
					{
						this->left = this->player.GetStone(x - 1).GetLeft();
						this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Left);
						this->player.RemoveStoneFromTheArray(this->player.GetStone(x - 1));
						pmove = true;
						cmove = false;
					}
					else
					{
						this->player.GetStone(x - 1).Reversal();
						reversep = false;
						if (this->player.GetStone(x - 1).GetLeft() == this->right)
						{
							this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Right);
							this->right = this->player.GetStone(x - 1).GetRight();
							this->player.GetStone(x - 1).Reversal();
							reversep = true;
							this->player.RemoveStoneFromTheArray(this->player.GetStone(x - 1));
							pmove = true;
							cmove = false;
						}
						else if (this->player.GetStone(x - 1).GetRight() == this->left)
						{
							this->gamePile.AddStoneToArray(this->player.GetStone(x - 1), Left);
							this->left = this->player.GetStone(x - 1).GetLeft();
							this->player.GetStone(x - 1).Reversal();
							reversep = true;
							this->player.RemoveStoneFromTheArray(this->player.GetStone(x - 1));
							pmove = true;
							cmove = false;
						}
						else
						{
							ToPrint = true;//if he chose wrong stone that not fit, so he must decide again another stone

						}
						if (!reversep)
							this->player.GetStone(x - 1).Reversal();
					}
				}
			}
		}
	}
}
