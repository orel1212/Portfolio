#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include "SafeQueue.cpp"
#include "MutexQueue.cpp"
#include "HelpArray.h"
#include <fstream>
#include <cstdlib>
extern "C"
{
	#include "util.h"
}
#define THREAD_MAX 5
static pthread_mutex_t console_mutex;
void * cast_Resolver(void * object);
void * cast_Requester(void * object);
class ThreadPool
{
private:
int noThreads;
SafeQueue * queue;
pthread_t * workers;
char  ** argv;
MutexQueue *  arr;
HelpArray * ha;
int * chooseIndex;
pthread_mutex_t choose_index_mutex;
public:
void joinRequesters();
ThreadPool(SafeQueue * queue,int noThreads,char  *argv[],HelpArray * ha);//requester
ThreadPool(SafeQueue * queue,int noThreads,MutexQueue *  arr,HelpArray * ha);//resolver
void run_Resolver();
void run_Requester();
~ThreadPool();
void Start(bool requester);
};
#endif
