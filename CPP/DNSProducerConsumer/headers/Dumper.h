#include "ThreadPool.cpp"
bool isVaildToOpen(char * path)
{
	ifstream ifs;
	ifs.open(path, ios::in);
	if (ifs.is_open()) 
	{
	    	ifs.close();
	    	return true;
	}
	else
	   	return false;	  
}
void * cast_Dumper(void * object);
class Dumper
{
	private:
	pthread_mutex_t output_mutex;
	MutexQueue *  arr;
	char * path;
	pthread_t * dumpers;
	int noDumpers;
	public:
	Dumper(MutexQueue *  arr,char * path,int noDumpers);
	void joinDumpers();
	~Dumper();
	void run_Dumper();
	void Start();
};
