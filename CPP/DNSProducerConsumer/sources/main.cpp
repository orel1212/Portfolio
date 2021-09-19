#include "Dumper.cpp"
int main(int argc, char  *argv[])
{
	  if(argc>2 && argc<13)
	  {
		  int * validFilesIndexes=new int[argc-2];//argc-1 is output file and 0 is ./ex5
		  int numofFiles=0;
		  
		  for(int i=1,j=0;i<argc-1;i++,j++)
		  {
			if (isVaildToOpen(argv[i]))
			 {
			    numofFiles++;
			    validFilesIndexes[j]=1;
			 }
			else
			    validFilesIndexes[j]=0;
	 	 }
	
		if(numofFiles>0)
		{
			char ** validFiles = new char*[numofFiles];
			for(int i=1,j=0;i<argc-1;i++)
			{
				if(validFilesIndexes[i-1]==1)
				{
					validFiles[j]=new char[ strlen(argv[i])+1];
					strcpy(validFiles[j],argv[i]);
					j++;
				}
				else
					cerr<< argv[i] << " is invalid" <<endl;
			}
			SafeQueue sq=SafeQueue();
			HelpArray *  ha=new HelpArray[numofFiles];
			MutexQueue arr=MutexQueue();
			ThreadPool res=ThreadPool(&sq,THREAD_MAX,&arr,ha);
			ThreadPool req=ThreadPool(&sq,numofFiles,validFiles,ha);
			Dumper dumpers=Dumper(&arr,argv[argc-1],1);
			pthread_mutex_init ( &console_mutex, NULL);
			res.Start(false);
			req.Start(true);
			req.joinRequesters();
			dumpers.Start();
			dumpers.joinDumpers();
			pthread_mutex_destroy(&console_mutex);
			if(ha)
				delete [] ha;
			if(validFiles)
			{
				for(int i=0;i<numofFiles;i++)
				{
					delete [] validFiles[i];
				}
				delete [] validFiles;
			}
		}
		else
			cerr << "all the files entered cannot open(not valid) " <<endl;
		if(validFilesIndexes)
				delete [] validFilesIndexes;
	}
	else if(argc<3)
		cerr << "need to insert at least 2 files,1 for requester and 1 for dumper(output)" <<endl;
	else
		cerr << "number of input files(num of requesters) bounded to 10" << endl;
	return 0;
}
