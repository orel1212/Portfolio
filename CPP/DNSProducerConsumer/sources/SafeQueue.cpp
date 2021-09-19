#include "SafeQueue.h"
SafeQueue::SafeQueue()
{
	pthread_mutex_init(&qmtx,0);
        pthread_cond_init(&wcond, 0);
	Front = NULL;
        Rear = NULL;
}
SafeQueue::~SafeQueue()
{
	pthread_mutex_destroy(&qmtx);
        pthread_cond_destroy(&wcond);
	if(Front)
		delete Front;
}
void SafeQueue::Enqueue(Task * task)
{
	
	pthread_mutex_lock(&qmtx);
	if(size==queueLimit)
	{
		 pthread_mutex_unlock(&qmtx);//before sleep free the lock, and after take it again.
		 usleep(rand() % 101);//sleep between 0-100 microseconds
		 pthread_mutex_lock(&qmtx);
	}
	if (Front == NULL)
        {
	    
            Front = new Node<Task>();
            Front->value = task;
            Front->next = NULL;
            Rear = Front;
        }
        else
        {
            Node<Task> * newNode = new Node<Task>();
            newNode->value = task;
            Rear->next = newNode;
            Rear = newNode;
        }
	size++;
        pthread_cond_broadcast(&wcond);//broadcast and not signal, to wakeup all waiting tasks
        pthread_mutex_unlock(&qmtx);
	
}
Task* SafeQueue::Dequeue()
{

	Task * t=NULL;
	pthread_mutex_lock(&qmtx);
	while (Front==NULL)
	{
            	pthread_cond_wait(&wcond, &qmtx);
	}
	if (Front != NULL)
        {
            Node<Task> * deleteNode = Front;
	    Front=Front->next;
            t=deleteNode->value;
            delete deleteNode;
	    size--;
        }
	pthread_mutex_unlock(&qmtx);
	return t;
	

}
bool SafeQueue::isEmpty()
{
	
	return Front==NULL;
}
