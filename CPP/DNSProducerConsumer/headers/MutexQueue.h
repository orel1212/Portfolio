#include "Node.cpp"
class MutexQueue
{
	private:
	Node<Task> * Front;
        Node<Task> * Rear;
	pthread_mutex_t mutex;
	int size;
	public:
	MutexQueue();
	~MutexQueue();
	void Enqueue(Task * task);
	Task * Dequeue();
	bool isEmpty();
	int getLength();
};
