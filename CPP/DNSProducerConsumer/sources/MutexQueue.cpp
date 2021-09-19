#include "MutexQueue.h"
MutexQueue::MutexQueue()
{
	Front = NULL;
        Rear = NULL;
	size=0;
	pthread_mutex_init(&mutex,0);
}
MutexQueue::~MutexQueue()
{
	if(Front)
		delete Front;
	pthread_mutex_destroy(&mutex);
}
void MutexQueue::Enqueue(Task * task)
{
	pthread_mutex_lock(&mutex);
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
	pthread_mutex_unlock(&mutex);
}
Task* MutexQueue::Dequeue()
{
	pthread_mutex_lock(&mutex);
	Task * t=NULL;
	if (Front != NULL)
        {
            Node<Task> * deleteNode = Front;
	    Front=Front->next;
            t=deleteNode->value;
            delete deleteNode;
	    size--;
        }
	pthread_mutex_unlock(&mutex);
	return t;
	

}
int MutexQueue::getLength()
{
	
	return size;
}
bool MutexQueue::isEmpty()
{
	
	return Front==NULL;
}
