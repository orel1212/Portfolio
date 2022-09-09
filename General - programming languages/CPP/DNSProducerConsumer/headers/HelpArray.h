#include "DevTask.h"
class HelpArray
{
	private:
	pthread_mutex_t qmtx;
    	pthread_cond_t wcond;
	public:
	HelpArray()
	{
		pthread_mutex_init(&qmtx,0);
        	pthread_cond_init(&wcond, 0);
	}
	~HelpArray()
	{
		pthread_mutex_destroy(&qmtx);
        	pthread_cond_destroy(&wcond);
	}
	void lockMutex()
	{
		pthread_mutex_lock(&qmtx);
	}
	void unlockMutex()
	{
		pthread_mutex_unlock(&qmtx);
	}
	void waitCondv()
	{
		pthread_cond_wait(&wcond, &qmtx);
	}
	void signalCondv()
	{
		pthread_cond_signal(&wcond);
	}
	
};
