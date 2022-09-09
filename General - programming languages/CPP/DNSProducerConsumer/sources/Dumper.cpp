#include "Dumper.h"
Dumper::Dumper(MutexQueue *  arr,char * path,int noDumpers):arr(arr),path(path),dumpers(new pthread_t[noDumpers]),noDumpers(noDumpers)
{
	pthread_mutex_init(&output_mutex,0);
}
Dumper::~Dumper()
{
	pthread_mutex_destroy(&output_mutex);
	if(dumpers)
		delete [] dumpers;
}
void * cast_Dumper(void * object)
{
	
	Dumper * obj=reinterpret_cast < Dumper * > ( object );		
	obj->run_Dumper();
	pthread_exit(NULL);
}
void Dumper::Start()
{
	pthread_attr_t attr;
	pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for(int i=0;i<noDumpers;++i)
	{
		int rc;
		
		rc = pthread_create(&(dumpers[i]), &attr, cast_Dumper, (void *)this);
		
		if (rc)
		{
			cerr <<"ERROR; return code from pthread_create() is "<< rc<<endl;
			exit(-1);
		}
	}

	pthread_attr_destroy(&attr);
}
void Dumper::joinDumpers()
{
	for (int i=0; i< noDumpers; i++)
	{
		pthread_join(dumpers[i], NULL);

	}
}
void Dumper::run_Dumper()
{
	
	
	while(arr->getLength()>0)
	{
		Task * t=arr->Dequeue();
		if(t!=NULL)
		{
			pthread_mutex_lock(&output_mutex);
			bool openned=false;
			ofstream file;
			if(isVaildToOpen(path))
			{
				file=ofstream(path,ofstream::out | ofstream::app);
				openned=true;
			}
			else
				cerr << "output file has invalid path" <<endl;
			if(openned)
			{
				int size=0;
				string * ips=t->getIPs(&size);
				file << t->getHostname() << "," ;
				if(size==0)
					file<<endl;
				else if(size==1)
					file << ips[0] << endl;
				else
				{
					for(int i=0;i<size-1;i++)
					{
						file << ips[0] << "," ;
					}
					file<<ips[size-1]<<endl;
				}
				file.close();
				delete t;
			}
			pthread_mutex_unlock(&output_mutex);
			
		}
	
	}
	   	
	pthread_exit(NULL);
	
}
