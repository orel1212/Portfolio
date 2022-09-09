#include "ThreadPool.h"
ThreadPool::ThreadPool(SafeQueue * queue,int noThreads,MutexQueue *  arr,HelpArray * ha):noThreads(noThreads),queue(queue),workers(new pthread_t[noThreads]),arr(arr),ha(ha)
{
	chooseIndex=new int[noThreads];
	for(int i=0;i<noThreads;i++)
	{
		chooseIndex[i]=0;
	}
	pthread_mutex_init(&choose_index_mutex,0);
}
ThreadPool::ThreadPool(SafeQueue * queue,int noThreads,char  *argv[],HelpArray * ha):noThreads(noThreads),queue(queue),workers(new pthread_t[noThreads]),argv(argv),ha(ha)
{
	chooseIndex=new int[noThreads];
	for(int i=0;i<noThreads;i++)
	{
		chooseIndex[i]=0;
	}
	pthread_mutex_init(&choose_index_mutex,0);
}
void ThreadPool::joinRequesters()
{
	for (int i=0; i< noThreads; ++i)
	{
		pthread_join(workers[i], NULL);

	}
}
void ThreadPool::Start(bool requester)
{
	pthread_attr_t attr;
	pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	int rc;
	for (int i=0; i< noThreads; ++i)
	{
		if(requester)
		{
			
			rc = pthread_create(&(workers[i]), &attr, cast_Requester, (void *)this);
		}
		else
		{
			rc = pthread_create(&(workers[i]), &attr, cast_Resolver, (void *)this);
		}
		if (rc)
		{
			cerr <<"ERROR; return code from pthread_create() is "<< rc<<endl;
			exit(-1);
		}

	}
	pthread_attr_destroy(&attr);
	
}
ThreadPool::~ThreadPool()
{
	if(chooseIndex)
		delete [] chooseIndex;
	pthread_mutex_destroy(&choose_index_mutex);
	if(workers)
		delete [] workers;
}
void * cast_Requester(void * object)
{
	
	ThreadPool * obj=reinterpret_cast < ThreadPool * > ( object );		
	obj->run_Requester();
	pthread_exit(NULL);
}
void * cast_Resolver(void * object)
{
	
	ThreadPool * obj=reinterpret_cast < ThreadPool * > ( object );		
	obj->run_Resolver();
	pthread_exit(NULL);
}

void ThreadPool::run_Requester()
{
	int index=noThreads;
	pthread_mutex_lock(&choose_index_mutex);
	for(int i=0;i<noThreads;i++)
	{
		if(chooseIndex[i]==0)
		{
			index=i;
			chooseIndex[i]=1;
			break;
		}
	}
	pthread_mutex_unlock(&choose_index_mutex);
	string line;
	ifstream infile(argv[index]);
	while(infile>>line)
	{
	    pthread_mutex_lock(&console_mutex);
            pthread_mutex_unlock(&console_mutex);
	    Task * t=new DevTask(index);
	    t->setHostname(line);
	    t->setID(index);
	    queue->Enqueue(t);
	    ha[index].lockMutex();
            ha[index].waitCondv();
	    ha[index].unlockMutex();
	    int size=0;
	    string * ips=t->getIPs(&size);
	    if(ips!=NULL)
	    {
	   	pthread_mutex_lock(&console_mutex);
	   	cout << t->getHostname() << "," ;
		if(size==1)
			cout << ips[0] << endl;
		else
		{
			for(int i=0;i<size-1;i++)
			{
				cout << ips[0] << "," ;
			}
			cout<<ips[size-1]<<endl;
		}
	   	 pthread_mutex_unlock(&console_mutex);
            }
	    delete t;
	   
	 }
	queue->doBroadCast();//to wake up the rest if some resolvers went to sleep
}
void ThreadPool::run_Resolver()
{
	int res_index=noThreads;
	pthread_mutex_lock(&choose_index_mutex);
	for(int i=0;i<noThreads;i++)
	{
		if(chooseIndex[i]==0)
		{
			res_index=i;
			chooseIndex[i]=1;
			break;
		}
	}
	pthread_mutex_unlock(&choose_index_mutex);
	while(true)
	{
		Task * t=queue->Dequeue();
		if(t!=NULL)
		{
			
			char ** ips=new char*[10];
			int addressCount=10;
			const char* hostname = t->getHostname().c_str();
			if(dnslookupAll(hostname, ips,10 ,&addressCount) == UTIL_FAILURE)
			{
				cerr<<"failed converting Hostname:" << t->getHostname() <<" to IP Address"<<endl;
				t->setIPs(NULL,0);
			}
			else
			{
				t->setIPs(ips,addressCount);
				
			}
			
			Task * arr_task=new DevTask(res_index);
			arr_task->setHostname(t->getHostname());
			arr_task->setIPs(ips,addressCount);
			arr->Enqueue(arr_task);
            		
			int index=t->getID();
			ha[index].lockMutex();
		    	ha[index].signalCondv();
		    	ha[index].unlockMutex();
			for(int i=0;i<addressCount;i++)
			{
				delete ips[i];
			}
			delete []ips;
			
		}
		else
			break;
		
	}
}
