#include <string>
#include <cstring>
#include <pthread.h>
#include <iostream>
using namespace std;
class Task
{
	
	public:
	virtual void setID(int id)=0;
	virtual int getID()=0;
	virtual void setIPs(char ** ips,int size)=0;
	//virtual void setIPs(string * ips,int size)=0;
	virtual void setHostname(string hostname)=0;
	virtual string *  getIPs(int * size)=0;
	virtual string getHostname()=0;
	virtual ~Task()
	{
	}
};


