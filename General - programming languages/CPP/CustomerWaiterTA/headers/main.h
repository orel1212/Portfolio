#include <string>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <wait.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#include <sys/shm.h>
#include <iomanip>
#include <ctime>
#include <sys/time.h>
#include "mysem.c"
using namespace std;
typedef struct {
    int Id;
    char Name[15];
    float Price;
    int TotalOrdered;
} menuRow;
typedef struct {
    int CustomerId;
    int ItemId;
    int Amount;
    bool Done;
} orderBoardRow;
int checkIfInteger(string str);
double getElapsedTime(timeval start);
void printMenu(menuRow * menu,int noDishes);
void fillMenu(menuRow * menu,int noDishes);
void PrintTotal(menuRow * menu,int noDishes);
string ItemIdToName(int chosenID);
void freeMemory(int menushmid,int ordershmid,int menurcshmid,int orderrcshmid);
int fillSemaphores(int * MenuQueue,int * OrderQueue,int * MenuReadAccess,int * OrderReadAccess,int * MenuResources,int * OrderResources,int * Output);
void fillOrderBoard(orderBoardRow * orderBoard,int noCustomers,int noDishes);
int memoryAllocationForOrderBoard(int noCustomers,int noDishes);
int checkIfValidInput(int argc,char * argv[]);
void printElpasedTime(double elpasedtime);
int memoryAllocationForMenu(int noDishes);
int memoryAllocationForMenuReadCount();
int memoryAllocationForOrderReadCount();
void AcquireMenuReading(int MenuQueue,int MenuReadAccess,int MenuResources,int * MenuReadCount);
void AcquireOrderBoardReading(int OrderQueue,int OrderReadAccess,int OrderResources,int * OrderReadCount);
void ReleaseMenuReading(int MenuReadAccess,int MenuResources,int * MenuReadCount);
void ReleaseOrderBoardReading(int OrderReadAccess,int OrderResources,int * OrderReadCount);
void AcquireMenuWriting(int MenuQueue,int MenuResources);
void AcquireOrderBoardWriting(int OrderQueue,int OrderResources);
void FreeSempahoresMemory(int MenuQueue,int OrderQueue,int MenuReadAccess,int OrderReadAccess,int MenuResources,int OrderResources,int Output);
