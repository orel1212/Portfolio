/*
It doesn't matter the order of the output lines, just they have to be the exactly ones that need to be printed.
for example thing that can occures:
 3.013 Customer ID 2: reads a menu about Salad (ordered, amount 1)
 9.015 Customer ID 2: reads a menu about Pie (ordered, amount 1)
 9.015 Waiter ID 0: performs the order of customer ID 2 (1 Salad)

it may looks like that the customer can order two times before a waiter finished his last order, but that's not the case,
because look at the time,it just printed the second line right before the third line, but the waiter performed the first order
before the customer made the second one.

*/
#include "main.h"
int main(int argc, char* argv[])
{
	if(checkIfValidInput(argc,argv)==1)
	{
		float simulationtime=stof(argv[1]);
		int numofdishes=stoi(argv[2]);
		int numofcustomers=stoi(argv[3]);
		int numofwaiters=stoi(argv[4]);
		srand (time(NULL));
		cout<<"=====Simluation arguments=====" << endl;
		cout << "Simulation time: "<< simulationtime << endl;
		cout << "Menu items count: "<< numofdishes << endl;
		cout << "Customers count: "<< numofcustomers << endl;
		cout << "Waiters count: "<< numofwaiters << endl;
		cout <<"=============================="<<endl;
		cout <<" 0.000 Main process ID " << getpid() << " start" << endl;
		int menushmid=memoryAllocationForMenu(numofdishes);
		int ordershmid=memoryAllocationForOrderBoard(numofcustomers,numofdishes);
		if(ordershmid==-1)
		{
			freeMemory(menushmid,-1,-1,-1);
			exit(1);
		}
		menuRow * menu=(menuRow *)shmat(menushmid, NULL, 0);
		orderBoardRow * orderBoard=(orderBoardRow *)shmat(ordershmid, NULL, 0);
		if(menu==(menuRow *)(-1) || orderBoard==(orderBoardRow *)(-1))
		{
			cerr<< "Shmat failed!" << endl;
			freeMemory(menushmid,ordershmid,-1,-1);
			exit(1);
		}
		fillMenu(menu,numofdishes);
		fillOrderBoard(orderBoard,numofcustomers,numofdishes);
		printMenu(menu,numofdishes);
		//create 2 readCounts: first for Menu, second for orderBoard
		int menurcshmid=memoryAllocationForMenuReadCount();
		if(menurcshmid==-1)
		{
			freeMemory(menushmid,ordershmid,-1,-1);
			exit(1);
		}
		int orderrcshmid=memoryAllocationForOrderReadCount();
		if(orderrcshmid==-1)
		{
			freeMemory(menushmid,ordershmid,menurcshmid,-1);
			exit(1);
		}
		int * MenuReadCount=(int *)shmat(menurcshmid, NULL, 0);
		int * OrderReadCount=(int *)shmat(orderrcshmid, NULL, 0);
		if(MenuReadCount==(int *)(-1) || OrderReadCount==(int *)(-1) )//sem init failed
		{

			cerr << "shmat failed" << endl;
			freeMemory(menushmid,ordershmid,menurcshmid,orderrcshmid);
			exit(1);
		}
		*(OrderReadCount)=0;
		*(MenuReadCount)=0;
		//create 6 semaphores: first 3 are queue,resource and readCountAccess for menu. 
		//the rest 3 are queue,resource and readCountAccess for orderBoard.
		int MenuQueue,OrderQueue,MenuReadAccess,OrderReadAccess,MenuResources,OrderResources,Output;
		if(fillSemaphores(&MenuQueue,&OrderQueue,&MenuReadAccess,&OrderReadAccess,&MenuResources,&OrderResources,&Output)==-1)//sem init failed
		{
			freeMemory(menushmid,ordershmid,menurcshmid,orderrcshmid);
			exit(1);
		}
		cout <<" 0.000 Main process start creating sub-process" << endl;
		struct timeval start;
		gettimeofday(&start, NULL);
		for(int i=0;i<numofcustomers;i++)
		{
			int pid=fork();
			if(pid==-1)
			{
				freeMemory(menushmid,ordershmid,menurcshmid,orderrcshmid);
				FreeSempahoresMemory(MenuQueue,OrderQueue,MenuReadAccess,OrderReadAccess,MenuResources,OrderResources,Output);
				cerr << "fork failed!" << endl;
				exit(1);
			}
			else if(pid==0)
			{
				int custID=i;
				while(true)
				{
					double elpasedtime=getElapsedTime(start);
					if(elpasedtime>simulationtime)
					{
						p(Output);
						printElpasedTime(elpasedtime);
						cout << " Customer ID " << custID << ": PID "<< getpid()<< " end work "; 
						cout << " PPID "<< getppid()  <<endl;
						v(Output);
						return 0;
					}
					//read menu
					AcquireMenuReading(MenuQueue,MenuReadAccess,MenuResources,MenuReadCount);
					srand(getpid() * time(NULL));
					float custread = (rand() %4) +3; 
					sleep(custread);
					ReleaseMenuReading(MenuReadAccess,MenuResources,MenuReadCount);
					//check if last order finished
					AcquireOrderBoardReading(OrderQueue,OrderReadAccess,OrderResources,OrderReadCount);
					bool flag=true;
					int nofFields=custID*numofdishes;
					int upToNextField=numofdishes+nofFields;
					for(int j=nofFields;j<upToNextField;j++)
					{
						if(orderBoard[j].Done==false)
						{
							flag=false;
							break;
						}
					}
					ReleaseOrderBoardReading(OrderReadAccess,OrderResources,OrderReadCount);
					if(!flag)
						continue;//back to a cuz there is a order that not completed yet by any waiter.
					else
					{
						srand(getpid() * time(NULL));
						int itemChosen = (rand() %numofdishes)+1;
						if((rand() %2)==0)
						{
							elpasedtime=getElapsedTime(start);
							if(elpasedtime<simulationtime)
							{
								p(Output);
								printElpasedTime(elpasedtime);
								cout << " Customer ID " << custID << ": reads a menu about "; 
								cout << ItemIdToName(itemChosen) << " (doesn't want to order)"<<endl;
								v(Output);
							}
							
						}
						else
						{

							//to make new order
							
							int amountChosen=(rand() %4)+1;
							AcquireOrderBoardWriting(OrderQueue,OrderResources);
							elpasedtime=getElapsedTime(start);
							if(elpasedtime<simulationtime)//double checking that he can perform an order now
							{
								int indexToUpdate=custID*numofdishes+(itemChosen-1);
								orderBoard[indexToUpdate].Amount=amountChosen;
								orderBoard[indexToUpdate].Done=false;
								v(OrderResources);
								p(Output);
								printElpasedTime(elpasedtime);
								cout << " Customer ID " << custID << ": reads a menu about "; 
								cout << ItemIdToName(itemChosen) << " (ordered, amount ";
								cout << amountChosen<<")"<<endl;
								v(Output);
							}
							else
								v(OrderResources);
							
						}
					}
				}
				
			}
			else
			{
				p(Output);
				double elpasedtime=getElapsedTime(start);
				printElpasedTime(elpasedtime);
				cout << " Customer " << i << ": created PID "<< pid << " PPID "<< getpid() <<endl;
				v(Output);
			}
		}
		for(int i=0;i<numofwaiters;i++)
		{
			int pid=fork();
			
			if(pid==-1)
			{
				freeMemory(menushmid,ordershmid,menurcshmid,orderrcshmid);
				FreeSempahoresMemory(MenuQueue,OrderQueue,MenuReadAccess,OrderReadAccess,MenuResources,OrderResources,Output);
				cerr << "fork failed!" << endl;
				exit(1);
			}
			else if(pid==0)
			{
				while(true)
				{
					
					AcquireOrderBoardReading(OrderQueue,OrderReadAccess,OrderResources,OrderReadCount);
					srand(getpid() * time(NULL));
					float waitercheck = (rand() %2) +1;
					sleep(waitercheck);
					int noFields=numofcustomers*numofdishes;
					int indexNotDone=-1;
					int amountToAdd=0;
					int indexOfDish=0;
					int customerID=-1;
					for(int j=0;j<noFields;j++)
					{
						if(orderBoard[j].Done==false)
						{
							indexNotDone=j;
							amountToAdd=orderBoard[j].Amount;
							indexOfDish=orderBoard[j].ItemId;
							customerID=orderBoard[j].CustomerId;
							break;
						}
					}
					ReleaseOrderBoardReading(OrderReadAccess,OrderResources,OrderReadCount);
					if(indexNotDone==-1)//if it true,it means that there is no order, so maybe he can end his job.
					{
						double elpasedtime=getElapsedTime(start);
						if(elpasedtime>simulationtime) 
						{
							p(Output);
							printElpasedTime(elpasedtime);
							cout << " Waiter ID " << numofwaiters-(i+1) << ": PID "<< getpid()<< " end work "; 
							cout << " PPID "<< getppid()  <<endl;
							v(Output);
							return 0;
						}
						else
							continue;
					}
					else
					{
						
						AcquireOrderBoardWriting(OrderQueue,OrderResources);
						if(orderBoard[indexNotDone].Done==false)//double checking dp
						{
							AcquireMenuWriting(MenuQueue,MenuResources);
							menu[indexOfDish-1].TotalOrdered+=amountToAdd;
							v(MenuResources);
							orderBoard[indexNotDone].Done=true;
							v(OrderResources);
							p(Output);
							double elpasedtime=getElapsedTime(start);
							printElpasedTime(elpasedtime);
							cout << " Waiter ID " << numofwaiters-(i+1) << ": performs the order of customer ID ";
							cout << customerID<< " ("<< amountToAdd << " "<< ItemIdToName(indexOfDish)<< ")" <<endl;
							v(Output);
						}
						else
							v(OrderResources);
						
					}
				}
				
				
			}
			else
			{
				p(Output);
				double elpasedtime=getElapsedTime(start);
				printElpasedTime(elpasedtime);
				cout << " Waiter " << numofwaiters-(i+1) << ": created PID "<< pid << " PPID "<< getpid() <<endl;
				v(Output);
			}
		}
		int noToWait=numofwaiters+numofcustomers;
		while(noToWait>0)
		{
			if(waitpid(-1, NULL, 0)>0)
				noToWait--;
		}
		printMenu(menu,numofdishes);
		PrintTotal(menu,numofdishes);
		double elaspedTime=getElapsedTime(start);
		cout << fixed << setprecision(3) << elaspedTime;
		cout << " Main ID "<< getpid() << " end work" << endl;
		cout << fixed << setprecision(3) << elaspedTime;
		cout << " End of simulation" << endl;
		freeMemory(menushmid,ordershmid,menurcshmid,orderrcshmid);
		FreeSempahoresMemory(MenuQueue,OrderQueue,MenuReadAccess,OrderReadAccess,MenuResources,OrderResources,Output);
		return 0;
	}
	exit(1);
	
}
int memoryAllocationForMenu(int noDishes)
{
	key_t key;
	int shmid;
	 /* make the key: */
	if ((key = ftok(".", 'v')) == -1) 
	{
		cerr<< "ftok failed!"<<endl;
		exit(1);
	}
	
	/* connect to (and possibly create) the segment: */
	int size=noDishes*sizeof(menuRow);
	if ((shmid = shmget(key, size, 0644 | IPC_CREAT)) == -1)
	{
		cerr<< "shmget failed!"<<endl;
		exit(1);
	}

	return shmid;
}
int memoryAllocationForOrderBoard(int noCustomers,int noDishes)
{
	key_t key;
	int shmid;
	 /* make the key: */
	if ((key = ftok(".", 'b')) == -1) 
	{
		cerr<< "ftok failed!"<<endl;
		return -1;
	}
	int size=noCustomers*noDishes*sizeof(orderBoardRow);
	/* connect to (and possibly create) the segment: */
	if ((shmid = shmget(key,size, 0644 | IPC_CREAT)) == -1)
	{
		cerr<< "shmget failed!"<<endl;
		return -1;
	}

	return shmid;
}
int memoryAllocationForOrderReadCount()
{
	key_t key;
	int shmid;
	 /* make the key: */
	if ((key = ftok(".", 'd')) == -1) 
	{
		cerr<< "ftok failed!"<<endl;
		return -1;
	}
	
	/* connect to (and possibly create) the segment: */
	if ((shmid = shmget(key, sizeof(shmid), 0644 | IPC_CREAT)) == -1)
	{
		cerr<< "shmget failed!"<<endl;
		return -1;
	}

	return shmid;
}
int memoryAllocationForMenuReadCount()
{
	key_t key;
	int shmid;
	 /* make the key: */
	if ((key = ftok(".", 'e')) == -1) 
	{
		cerr<< "ftok failed!"<<endl;
		return -1;
	}
	
	/* connect to (and possibly create) the segment: */
	if ((shmid = shmget(key, sizeof(shmid), 0644 | IPC_CREAT)) == -1)
	{
		cerr<< "shmget failed!"<<endl;
		return -1;
	}

	return shmid;
}
int checkIfValidInput(int argc,char * argv[])
{
	int flag=1;
	if(argc!=5)//argv[0] is ./ex3
	{
		cerr <<	"Input Argument are not valid" << endl;
		return 0;
	}
	else
	{
		for(int i=1;i<argc;i++)
		{
			string temp=string(argv[i]);
			bool check=checkIfInteger(temp);
			if(check==0)
			{
				cerr<<"The Argument No#" << i << " is not integer"<< endl;
				return 0;
			}
			else
			{
				switch(i)
				{
				  case 1://simulation time
					{
						float simtime=stof(temp);
						if(simtime<1 || simtime>30)
						{
							cerr<<"Simulation time must be bounded between 1 to 30"<< endl;
							flag=0;
						}
					}
				        break;
				  case 2://num of dishes
					{
						int nodishes=stoi(temp);
						if(nodishes<5 || nodishes>10)
						{
							cerr<<"Num of Dishes must be bounded between 5 to 10"<< endl;
							flag=0;
						}
					}
				        break;
				  case 3://num of customers
					{
						int nocustomers=stoi(temp);
						if(nocustomers<1 || nocustomers>10)
						{
							cerr<<"Num of Customers must be bounded between 1 to 10"<< endl;
							flag=0;
						}
					}
				        break;
				  case 4://num of waiters
					{
						int nowaiters=stoi(temp);
						if(nowaiters<1 || nowaiters>3)
						{
							cerr<<"Num of Waiters must be bounded between 1 to 3"<< endl;
							flag=0;
						}
					}
				        break;
				}
			}
		}
	}
	return flag;
}
int checkIfInteger(string str)
{
	int size=str.length();
	for(int i=0;i<size;i++)
	{
		if(str[i]<'0' || str[i]>'9')
			return 0; 
	}
	return 1;
}
double getElapsedTime(timeval start)
{
	struct timeval curr;
	gettimeofday(&curr, NULL);
	return ((curr.tv_sec  - start.tv_sec) * 1000000u + curr.tv_usec - start.tv_usec) / 1.e6;
}
void printMenu(menuRow * menu,int noDishes)
{
	cout <<"==========Menu list==========="<<endl;
	cout << "Id Name      Price  Orders" << endl;
	for(int i=0;i<noDishes;i++)
	{
		cout << menu[i].Id ;
		if(i==9)
			cout<< " ";
		else
			cout<< "  ";
		cout<< menu[i].Name ;
		string spacesAdd(menu[i].Name);
		int numofspaces=9-spacesAdd.length();
		string spaces(" ");
		for(int j=0;j<numofspaces;j++)
			spaces+=" ";
		cout << spaces ;
		cout << fixed << setprecision(2) << menu[i].Price ;
		if(i==0 || i==2 || i==7 || i==9)
			cout << "  " ;
		else
			cout << "   " ;
		cout << menu[i].TotalOrdered << endl;
	}
	cout <<"=============================="<<endl;
}
void fillMenu(menuRow * menu,int noDishes)
{
		for(int i=0;i<noDishes;i++)
		{
			menu[i].Id=i+1;
			menu[i].TotalOrdered=0;
		}
		strcpy(menu[0].Name,"Pizza");
		menu[0].Price=10.00;
		strcpy(menu[1].Name,"Salad");
		menu[1].Price=8.00;
		strcpy(menu[2].Name,"Hamburger");
		menu[2].Price=12.00;
		strcpy(menu[3].Name,"Spaghetti");
		menu[3].Price=9.00;
		strcpy(menu[4].Name,"Pie");
		menu[4].Price=9.50;
		if(noDishes>5)
		{
			strcpy(menu[5].Name,"Milkshake");
			menu[5].Price=6.00;
			if(noDishes>6)
			{
				strcpy(menu[6].Name,"Chicken");
				menu[6].Price=6.00;
				if(noDishes>7)
				{
					strcpy(menu[7].Name,"Icecream");
					menu[7].Price=10.00;
					if(noDishes>8)
					{
						strcpy(menu[8].Name,"Steak");
						menu[8].Price=5.00;
						if(noDishes>9)
						{
							strcpy(menu[9].Name,"Kebab");
							menu[9].Price=12.00;
						}
					}
				}
			}
		}
}
void fillOrderBoard(orderBoardRow * orderBoard,int noCustomers,int noDishes)
{
		int noFields=noCustomers*noDishes;
		for(int i=0,CustID=0,ItemID=1;i<noFields;i++)
		{
			orderBoard[i].CustomerId=CustID;
			orderBoard[i].ItemId=ItemID;
			orderBoard[i].Amount=0;
			orderBoard[i].Done=true;
			if(i>0 && (i+1)%noDishes==0)
			{
				CustID++;
				ItemID=0;//right after the if it will be 1 again,it need to be resetted after noDishes items to 1.
			}
			ItemID++;
		}
}
int fillSemaphores(int * MenuQueue,int * OrderQueue,int * MenuReadAccess,int * OrderReadAccess,int * MenuResources,int * OrderResources,int * Output)
{
	key_t semkey = ftok(".",1);
	if(semkey==-1)
	{
		cerr << "ftok failed! " << endl;
		return -1;
	}
	if (((*MenuQueue)=initsem(semkey,1))<0) /*initialize semaphore*/
	{
		cerr << "sempahore initialization failed! " << endl;
		return -1;
	}
	semkey = ftok(".",2);
	if(semkey==-1)
	{
		cerr << "ftok failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),-1,-1,-1,-1,-1,-1);
		return -1;
	}
	if (((*OrderQueue)=initsem(semkey,1))<0) /*initialize semaphore*/
	{
		cerr << "sempahore initialization failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),-1,-1,-1,-1,-1,-1);
		return -1;
	}
	semkey = ftok(".",3);
	if(semkey==-1)
	{
		cerr << "ftok failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),(*OrderQueue),-1,-1,-1,-1,-1);
		return -1;
	}
	if (((*MenuReadAccess)=initsem(semkey,1))<0) /*initialize semaphore*/
	{
		cerr << "sempahore initialization failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),(*OrderQueue),-1,-1,-1,-1,-1);
		return -1;
	}
	semkey = ftok(".",4);
	if(semkey==-1)
	{
		cerr << "ftok failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),(*OrderQueue),(*MenuReadAccess),-1,-1,-1,-1);
		return -1;
	}
	if (((*OrderReadAccess)=initsem(semkey,1))<0) /*initialize semaphore*/
	{
		cerr << "sempahore initialization failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),(*OrderQueue),(*MenuReadAccess),-1,-1,-1,-1);
		return -1;
	}
	semkey = ftok(".",5);
	if(semkey==-1)
	{
		cerr << "ftok failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),(*OrderQueue),(*MenuReadAccess),(*OrderReadAccess),-1,-1,-1);
		return -1;
	}
	if (((*MenuResources)=initsem(semkey,1))<0) /*initialize semaphore*/
	{
		cerr << "sempahore initialization failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),(*OrderQueue),(*MenuReadAccess),(*OrderReadAccess),-1,-1,-1);
		return -1;
	}
	semkey = ftok(".",6);
	if(semkey==-1)
	{
		cerr << "ftok failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),(*OrderQueue),(*MenuReadAccess),(*OrderReadAccess),(*MenuResources),-1,-1);
		return -1;
	}
	if (((*OrderResources)=initsem(semkey,1))<0) /*initialize semaphore*/
	{
		cerr << "sempahore initialization failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),(*OrderQueue),(*MenuReadAccess),(*OrderReadAccess),(*MenuResources),-1,-1);
		return -1;
	}
	semkey = ftok(".",7);
	if(semkey==-1)
	{
		cerr << "ftok failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),(*OrderQueue),(*MenuReadAccess),(*OrderReadAccess),(*MenuResources),(*OrderResources),-1);
		return -1;
	}
	if (((*Output)=initsem(semkey,1))<0) /*initialize semaphore*/
	{
		cerr << "sempahore initialization failed! " << endl;
		FreeSempahoresMemory((*MenuQueue),(*OrderQueue),(*MenuReadAccess),(*OrderReadAccess),(*MenuResources),(*OrderResources),-1);
		return -1;
	}
	return 0;
}
void PrintTotal(menuRow * menu,int noDishes)
{
	int noOrders=0;
	float total=0.0;
	for(int i=0;i<noDishes;i++)
	{
		noOrders+=menu[i].TotalOrdered;
		total+=menu[i].Price*menu[i].TotalOrdered;
	}
	cout << "Total orders " << noOrders <<", for an amount " << fixed << setprecision(2) << total << " NIL" << endl;
}
void freeMemory(int menushmid,int ordershmid,int menurcshmid,int orderrcshmid)
{
	if(menushmid>-1)
		shmctl(menushmid, IPC_RMID, 0);
	if(ordershmid>-1)
		shmctl(ordershmid, IPC_RMID, 0);
	if(menurcshmid>-1)
		shmctl(menurcshmid, IPC_RMID, 0);
	if(orderrcshmid>-1)
		shmctl(orderrcshmid, IPC_RMID, 0);
}
string ItemIdToName(int chosenID)
{
	if(chosenID==1)
		return string("Pizza");
	else if(chosenID==2)
		return string("Salad");
	else if(chosenID==3)
		return string("Hamburger");
	else if(chosenID==4)
		return string("Spaghetti");
	else if(chosenID==5)
		return string("Pie");
	else if(chosenID==6)
		return string("Milkshake");
	else if(chosenID==7)
		return string("Chicken");
	else if(chosenID==8)
		return string("Icecream");
	else if(chosenID==9)
		return string("Steak");
	else
		return string("Kebab");
}
void printElpasedTime(double elpasedtime)
{
	if(elpasedtime<10)//to move the number as in the example
	{
		cout << " " ;
	}
	cout << fixed << setprecision(3) << elpasedtime;
}
void AcquireMenuReading(int MenuQueue,int MenuReadAccess,int MenuResources,int * MenuReadCount)
{
	p(MenuQueue);
	p(MenuReadAccess);
	if(*(MenuReadCount)==0)
	{
		p(MenuResources);	
	}
	*(MenuReadCount)+=1;
	v(MenuQueue);
	v(MenuReadAccess);
}
void AcquireOrderBoardReading(int OrderQueue,int OrderReadAccess,int OrderResources,int * OrderReadCount)
{
	p(OrderQueue);
	p(OrderReadAccess);
	if(*(OrderReadCount)==0)
	{
		p(OrderResources);	
	}
	*(OrderReadCount)+=1;
	v(OrderQueue);
	v(OrderReadAccess);	
}
void ReleaseMenuReading(int MenuReadAccess,int MenuResources,int * MenuReadCount)
{
	p(MenuReadAccess);
	*(MenuReadCount)-=1;
	if(*(MenuReadCount)==0)
	{
		v(MenuResources);	
	}
	v(MenuReadAccess);
}
void ReleaseOrderBoardReading(int OrderReadAccess,int OrderResources,int * OrderReadCount)
{
	p(OrderReadAccess);
	*(OrderReadCount)-=1;
	if(*(OrderReadCount)==0)
	{
		v(OrderResources);	
	}
	v(OrderReadAccess);
}
void AcquireMenuWriting(int MenuQueue,int MenuResources)
{
	p(MenuQueue);
	p(MenuResources);
	v(MenuQueue);
}
void AcquireOrderBoardWriting(int OrderQueue,int OrderResources)
{
	p(OrderQueue);
	p(OrderResources);
	v(OrderQueue);
}
void FreeSempahoresMemory(int MenuQueue,int OrderQueue,int MenuReadAccess,int OrderReadAccess,int MenuResources,int OrderResources,int Output)
{
	if(MenuQueue>-1)
	{
		semctl(MenuQueue, IPC_RMID, 0);
	}
	if(OrderQueue>-1)
	{
		semctl(OrderQueue, IPC_RMID, 0);
	}
	if(MenuReadAccess>-1)
	{
		semctl(MenuReadAccess, IPC_RMID, 0);
	}
	if(OrderReadAccess>-1)
	{
		semctl(OrderReadAccess, IPC_RMID, 0);
	}
	if(MenuResources>-1)
	{
		semctl(MenuResources, IPC_RMID, 0);
	}
	if(OrderResources>-1)
	{
		semctl(OrderResources, IPC_RMID, 0);
	}
	if(Output>-1)
	{
		semctl(Output, IPC_RMID, 0);
	}
}
