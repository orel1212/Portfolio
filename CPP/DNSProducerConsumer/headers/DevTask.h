#ifndef DEV_TASK_H
#define DEV_TASK_H
#include "Task.h"
class DevTask: public Task
{
private:
string * ips;
string hostName;
int size;
int id;
public:
DevTask(int id)
{
ips=NULL;
size=0;
this->id=id;
}
void setID(int id)
{
this->id=id;
}
int getID()
{
return id;
}
void setIPs(char ** ips,int size)
{
	if(this->ips)
		delete [] this->ips;
	this->size=size;
	if(size>0)
	{
		this->ips=new string[size];
		for(int i=0;i<size;i++)
		{
			this->ips[i]=string(ips[i]);
		}
	}
	else
		ips=NULL;
}
string *  getIPs(int * size)
{
	*(size)=this->size;
	return this->ips;
}
void setHostname(string hostname)
{
this->hostName=hostname;
}
string getHostname()
{
return this->hostName;
}
~DevTask()
{

}
};
#endif
