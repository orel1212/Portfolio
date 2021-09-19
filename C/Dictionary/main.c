#define _CRT_SECURE_NO_WARNINGS
#define WORD 81
#define DEFENITION 201
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
void FreeMemory(char ***ptr, int size);//function that free the dynamic memory
void PrintWAndD(char ***ptr, int wordindex);//function that print the word and the defenitions to the word
int BinarySearch(char ***ptr, int low, int high, int size, char *str);//funtion that search as the best way to do recursive search-binary search
void MoveBack(char ***ptr, int index, int size);//function that arrives the next pointers to the current
void SimiliarWords(char ***ptr, int *size); //function that delete similiar words
void LowerToUpper(char *ptr, int size);//function that make lowercase to uppercase and opposite
void Lexicography(char ***ptr, int size);//function that make the dictionary as lexicograhpy
void MemoryProblem(char ***ptr, int size);//function that finish the program if there's a memory problem
char ** InsertValues(char ***ptr, int size);//function that insert to each ptr[i] modified ptr**
void Dictionary();
int main()
{
	Dictionary();//dictionary is include the ***pointer 
	return;
}
void Dictionary()
{
	char str[WORD];
	char ***ptr=NULL;
	int size,i,j;
	printf("Enter how many words do you want to insert to the dictionary\n");
	scanf_s("%d", &size);
	ptr = (char***)malloc((size)*sizeof(char**));//dynamic memory to ***ptr
	if (ptr != NULL)//if ptr isn't null we can continue
	{
		for (i = 0; i < size; i++)
		{

			ptr[i]=InsertValues(&ptr[i],size);//we insert each place in the ***ptr a **ptr when we send to Insertvalue the place, and insertvalue return ready **ptr
		}
		for (i = 0; i < size; i++)
		{
			for (j = 0; ptr[i][j] != NULL; j++)
			{
				
				LowerToUpper(ptr[i][j], 0);//modify every word and definition as rules, for example:first letter must be uppercase.
			}
		}
		Lexicography(ptr, size);//to make the dict as lexicograhpy
		SimiliarWords(ptr, &size);//to delete similiar words
		printf("The Dictionary is ready.\n");
		printf("If you want to search a word in the dictionary-insert the word.\n");
		printf("When you finished to search words in the dictionary -insert exit\n");
		do//until the user enters exit, he can continue search for words
		{
			gets(str);
			if (strcmp(str, "exit") != 0)//if the string the user enters isn't exit
			{
				LowerToUpper(str, 0);
				if (BinarySearch(ptr, 0, size, size, str) == -1)//if after binary search -1 is returned, it mean that the word he entered not exist
					printf("Unknown word\n");
				else
					PrintWAndD(ptr, BinarySearch(ptr, 0, size, size, str));//if the word exist, printWandD will print the word and the defenitions, and how many defenitions
			}
		} while (strcmp(str, "exit") != 0);
		FreeMemory(ptr, size);//free the dynamic memory that we used
	}
	else
		MemoryProblem(ptr,size);//if there's a memory problem ,the program is closed
	

}
char ** InsertValues(char ***ptr, int size)
{
	char ch[WORD], chd[DEFENITION];
	int sizeod, j;
		printf("Enter how many defenitions you want to enter to the dictionary\n");
		scanf_s("%d", &sizeod);
		getchar();
		ptr = (char**)malloc((sizeod + 2)*sizeof(char*));//+2 because we need '\0' and 1 more field for the word, besides the defenitions 
		if (ptr != NULL)
		{
			printf("Please enter a word that must be maximum 80 chars without spaces\n");
			gets(ch);
			ptr[0] = (char*)malloc((strlen(ch) + 1)*sizeof(char));//we need +1 to \0
			if (ptr[0] != NULL)
			{
				strcpy(ptr[0], ch);//we can insert it into the ptr** at the first place
				for (j = 1; j <= sizeod;j++)
				{
					printf("please enter the %d definition\n", j);
					gets(chd);
					ptr[j] = (char*)malloc((strlen(chd) + 1)*sizeof(char));//to every def we must make a dynamic array
					strcpy(ptr[j], chd); //we can insert it into the ptr** at the j place
					
				}
				ptr[j] = '\0';//after the for is d,we enters '\0' to know after where to stop to print the defs.
			}
			else
				MemoryProblem(ptr,size);
		}
		else
			MemoryProblem(ptr, size);
		return ptr;//after the ptr** is ready we can return it to the dictionary()
}
void MemoryProblem(char ***ptr,int size)
{
	printf("There is a problem with the memory\n");
	FreeMemory(ptr, size);//to free the memory after the user entered exit
	exit(1);//we out from the program if there's a memory problem
}

void Lexicography(char ***ptr, int size)
{
	int i;
	char * help;
	if (size== 0)//stop term
		return;
	for (i = 0; i<size- 1; i++)
	{
		if (strcmp(ptr[i + 1][0], ptr[i][0]) == -1)//sort like bubblesort
		{
			help = ptr[i + 1];
			ptr[i + 1] = ptr[i];
			ptr[i] = help;
		}
	}
	Lexicography(ptr, size - 1);//recurisve until size 0
}

void LowerToUpper(char *ptr, int size)
{

	if (ptr[size] == '\0')//if we are standing in '\0' so we finish, stop term
		return;
	if (size==0)
		if (ptr[size]<='z' && ptr[size]>='a')//if the first place in the dynamic array is lowercase, we must make it uppercase
			ptr[size] = ptr[size] - 'a' + 'A';
	if (size>0)//if we're not in the first place of the array
		if (ptr[size] >= 'A' && ptr[size] <= 'Z')//we must change every uppercase to lowercase
			ptr[size] -= ('A' - 'a');
	if (ptr[size + 1] == ' ' && ptr[size] == '.')//if we're standing on '.' and the next is ' ', so the next letter must be uppercase
	{
		size += 2;//to advance to the next letter
		if (ptr[size]<='z' && ptr[size]>='a')
			ptr[size] = ptr[size] - 'a' + 'A';
	}
	if (ptr[size]==' ')//if we're standing on ' ' only we can continue and the next letters must be lowercase
	{
		size++;
		
	}
	LowerToUpper(ptr, size + 1);

}
void SimiliarWords(char ***ptr, int *size)
{
	int i, j,k,sent;
	for (i = 0; i < *size-1; i++)//to check every word until ptr[size-1] cuz we don't have to compare it with the next word,cuz there's not next word
	{
		for (j = i+1; j < *size; j++)//we stand from the next word,not to check the same word if i==j
		{
			if (strcmp(ptr[i][0], ptr[j][0]) == 0)//if there's same word
			{
				for (k = 0; ptr[j][k] != NULL; k++)//we must free the current ** place in the *** cell at the array
					free(ptr[j][k]);
					free(ptr[j][k]);//free the '\0' cell
				free(ptr[j]);
				sent = *size;
				MoveBack(ptr, j, sent);//to take back all the next words, so we will not lose them
					*size -= 1;//to make the array smaller by 1
					ptr = (char***)realloc(ptr, (*size)*sizeof(char**));//to make the array smaller by 1
					if (ptr == NULL)
						MemoryProblem(ptr, size);

			}
		}
	}
}
void MoveBack(char ***ptr, int index, int size)
{
	int i;
	if (index == size - 2)
		ptr[index] = ptr[index + 1];
	else
	{
		while (index<(size-1))//to take back all the next words, so we will not lose them
		{
			ptr[index] = ptr[index + 1];
			index++;
		}
			
	}

}
int BinarySearch(char ***ptr, int low, int high, int size, char *str)
{
	int result, midd;
	midd = (low + high) / 2;//split the array to twosides every time,and the binarysearch search only at the size that the word must be in there, for example: if the word is 'X' , then it cannot be in 'A' to 'L' , it must be in 'M' to 'Z'
	if (low > high || midd>=size)//if low is higher than high or midd is equal or higher than size then there's no word in the array as the user entered to search
		return -1;
	result = strcmp(ptr[midd][0], str);//put inside result what strcmp return
	if (result< 0)//if the word he entered is must be after the left size of the array,so we must sent recurisve midd+1 in low cuz he must be in the right side
		return BinarySearch(ptr, midd + 1, high, size, str);
	else if (result > 0)//if the word he entered is must be at the left size of the array,so we must sent recurisve and high must be until mid-1, cuz he must be in the left side
		return BinarySearch(ptr, low, midd - 1, size, str);
	else if (result == 0)
		return midd;
}

void PrintWAndD(char ***ptr, int wordindex)
{
	int j, count;
	for (count = 0; ptr[wordindex][count] != '\0'; count++);//to count how many defenitions we have
	printf("There are %d defenitions to the word you chose\n", count);
	printf("The word is: ");
	puts(ptr[wordindex][0]);//to print the word
	printf("\n");
	j = 1;//to start printing from defenitions
	while (ptr[wordindex][j] != '\0')//if we're not reached yet '\0'
	{
		printf("The %d defenition is :", j);
		puts(ptr[wordindex][j]);//print the defenition
		printf("\n");
		j++;
	}
}
void FreeMemory(char ***ptr, int size)
{
	int i, j;
	for (i = 0; i < size; i++)
	{
		for (j = 0; ptr[i][j] != NULL; j++)
		{
			free(ptr[i][j]);//we free firstly every word and defenition at the **
		}
		free(ptr[i][j]);//we free the '\0' cell at the ** ptr
		free(ptr[i]);//then free the place in the ***
	}
	free(ptr);//then free the whole ***array.
}