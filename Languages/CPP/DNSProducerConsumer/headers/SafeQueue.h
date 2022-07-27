#define queueLimit 1000
#include <unistd.h> // for sleeping only: usleep(microseconds); (as asked by Daniel Hankin)
#include "Node.cpp"
class SafeQueue
{
	private:
	pthread_mutex_t qmtx;
    	pthread_cond_t wcond;
	Node<Task> * Front;
        Node<Task> * Rear;
	int size;
	public:
	SafeQueue();
	~SafeQueue();
	void Enqueue(Task * task);
	Task * Dequeue();
	bool isEmpty();
	void doBroadCast()
	{
		pthread_mutex_lock(&qmtx);
        	pthread_cond_broadcast(&wcond);//broadcast and not signal, to wakeup all waiting tasks
        	pthread_mutex_unlock(&qmtx);
	}
};
