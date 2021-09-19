#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define ID 9
#define NameSIZE 11
#define DescSIZE 201
#define NumOfHobbies 4
#define SizeOfPassword 7
enum Option { option1 = 1, option2 };
enum SOF { Success = 0, Fail };
typedef enum { hobbie1, hobbie2, hobbie3, hobbie4 } hobbie;
typedef struct
{
	char* id;
	char* fname;
	char* lname;
	int age;
	char gender;
	char* username;
	char* password;
	char* decription;
	char** hobbies;
} Newmember;

typedef struct
{
	Newmember* community;
} community;
void MainMenu();//that funcitions print the mainmenu to the screen
void FreeMember(Newmember* member);//free the memory of a member if there's a problem with the memory
void FreeMemory(community* community, int size);//free the memory of the community
void MainMenuOptions(community community, int index, int size);//that function controls the mainmenu options
void EnterMenuOptions();//that function controls all the program.
void CreateNewMember(community* community, int* size);//that functions create new member
void FindMatch(community community, int index, int size);//that function try to find a match between the user to another people
void PrintMatch(community community, int index);//if there's a match, that function print the details of the match
void ReadInformation(community* community, int* size, int* index);//that function read information from text 
int main()
{
	printf("Dating system! \n");
	EnterMenuOptions();
	return 0;
}
void EnterMenu()
{
	printf("1. New Member \n");
	printf("2. LogIn \n");
}
void MainMenu()
{
	printf("1. Search Matching \n");
	printf("2. Exit \n");
}
void EnterMenuOptions()
{
	community community;//community type
	FILE* file;//a pointer to the file
	char username[NameSIZE], password[SizeOfPassword];
	int i = 0, option, size = 1, j, flagname = Fail, flagpassword = Fail;
	community.community = (Newmember*)malloc(size * sizeof(Newmember));//create a new community that holds all the members
	if (community.community == NULL)//check if there's a memmory problem
	{
		printf("There is a problem with the dynamic memory \n");
		exit(1);
	}
	ReadInformation(&community, &size, &i);//read all information from text
	do
	{
		EnterMenu();
		scanf("%d", &option);
		if (option == option1)//if the user chose new member option
		{
			//if (size>1)
				//community.community = (Newmember*)realloc(community.community, size*sizeof(Newmember));
			if (community.community == NULL)//check if there's memory problem
			{
				printf("There is a problem with the dynamic memory \n");
				FreeMemory(&community, size);
				exit(1);
			}
			CreateNewMember(&community, &size);//create new member
			printf("Hi %s, welcome to the Dating site\n", community.community[i].fname);
			MainMenuOptions(community, i, size);//we send i-1,size-1 cuz we talking about the current person, and not about the next person ,that not inserted yet
		}
		else if (option == option2)//if the user chose login
		{
			getchar();
			do
			{
				printf("Please enter your username, a word without spaces\n");
				gets(username);
				printf("Please enter your password , without spaces\n");
				gets(password);
				for (j = 0; j < size; j++)
				{
					if (strcmp(community.community[j].username, username) == Success)//if the user's username input is same as in the data base
						flagname = Success;//flagname=success(0) if the user entered a user name that valid in the database
					if (strcmp(community.community[j].password, password) == Success)//if the user's password input is same as in the data base
						flagpassword = Success;//flagpassword=success(0) if the user entered a password that valid in the database
					if (flagname == Success && flagpassword == Success)//if both flags are ==success, that means that he entered correct user and passowrd
						break;
				}
				if (flagpassword != Success || flagname != Success)//if even one of them isn't success, that means that he entered wrong input
					printf("user name or password do not exist in the system, please try again\n");
				else
				{
					printf("Hello %s %s \n", community.community[j].fname, community.community[j].lname);//he can go into the system, because he entered correct user and password
					MainMenuOptions(community, j, size);
				}
			} while (flagpassword != Success || flagname != Success);//do-while until the user entered correct user and password

		}
		else
			printf("Bad choice, please enter again.\n");//if the user chose another option from 1 to 2
	} while (option != option2 && option != option1);//do-while until the user enter 1 or 2
	file = fopen("input.txt", "w");//open the file as mode 'w', so all the information in the file will be deleted, and we will rewrite them with the updates
	if (file == NULL)
	{
		printf("There is a problem with opening the file \n");
		exit(1);
	}
	for (i = 0; i < size; i++)//for loop for every member(community[i])
	{
		fprintf(file, "%s ", community.community[i].id);//print to the file the id
		fprintf(file, "%s ", community.community[i].fname);//print to the file the first name 
		fprintf(file, "%s ", community.community[i].lname);//print to the file the last name
		fprintf(file, "%d ", community.community[i].age);//print to the file the age
		fprintf(file, "%c ", community.community[i].gender);//print to the file the gender
		fprintf(file, "%s ", community.community[i].username);//print to the file the username
		fprintf(file, "%s ", community.community[i].password);//print to the file the password
		fprintf(file, "%s ; ", community.community[i].decription);//print to the file the description
		for (j = 0; j < NumOfHobbies; j++)//for loop to print all 4 hobbies
		{
			fprintf(file, "%s ", community.community[i].hobbies[j]);//print to the file the hobbie[j]
		}
		fprintf(file, "\n");
	}
	fclose(file);//after we finish to write in the file, we must close him	
	FreeMemory(&community, size);
}
void MainMenuOptions(community community, int index, int size)
{
	int option;
	do
	{
		MainMenu();
		scanf("%d", &option);
		if (option == option1)//if the user want to find match
		{
			FindMatch(community, index, size);
		}
		else if (option == option2)//if the user want to exit the system
		{
			//if the user press option2(2) , it doesn't do anything here, but after it comeback to EntermenuOptions() , it write to the file and free the memory.
			//to do a duplicate free my cause problems, so just after it came back to EnterMenuOptions(),that free the memory
		}
		else
			printf("Bad choice, please enter again.\n");//if the user entered a input that isn't 1 or 2
	} while (option != option1 && option != option2);//do while loop until the user enter 1 or 2
}
void CreateNewMember(community* community, int* size)
{
	int i;
	char name[NameSIZE], desc[DescSIZE], hobbie[NameSIZE], password[SizeOfPassword];
	Newmember member;
	getchar();
	printf("Please enter your ID Number \n");
	member.id = (char*)malloc((ID + 1) * sizeof(char));//to create 9 fields for every digit in the id, and one for '\0'
	if (member.id == NULL)
	{
		printf("There is a problem with the dynamic memory \n");
		exit(1);
	}
	gets(member.id);//insert the id into the id field, in newmember struct
	printf("Enter your First name, maximum 10 chars\n");
	gets(name);
	member.fname = (char*)malloc((strlen(name) + 1) * sizeof(char));//we create a place for first name with exaclty fields we need
	if (member.fname == NULL)
	{
		printf("There is a problem with the dynamic memory \n");
		FreeMember(&member);
		exit(1);
	}
	strcpy(member.fname, name);//insert the firstname into the fname field, in newmember struct
	printf("Enter your Last name, maximum 10 chars\n");
	gets(name);
	member.lname = (char*)malloc((strlen(name) + 1) * sizeof(char));//we create a place for last name with exaclty fields we need
	if (member.lname == NULL)
	{
		printf("There is a problem with the dynamic memory \n");
		FreeMember(&member);
		exit(1);
	}
	strcpy(member.lname, name);//insert the lastname into the lname field, in newmember struct
	printf("Please enter your age between 18 to 100 \n");
	scanf("%d", &(member.age));//insert the age into the age field, in newmember struct
	getchar();
	printf("Please enter your gender - F if female, M if male \n");
	scanf("%c", &(member.gender));//insert the gender into the gender field, in newmember struct
	getchar();
	printf("Please enter your username, between 5-10 chars, first char is a letter \n");
	gets(name);
	member.username = (char*)malloc((strlen(name) + 1) * sizeof(char));//we create a place for user name with exaclty fields we need
	if (member.username == NULL)
	{
		printf("There is a problem with the dynamic memory \n");
		FreeMember(&member);
		exit(1);
	}
	strcpy(member.username, name);//insert the username into the username field, in newmember struct
	printf("Please enter your password, maximum 6 chars \n");
	gets(password);
	member.password = (char*)malloc((strlen(password) + 1) * sizeof(char));//we create a place for password with exaclty fields we need
	if (member.password == NULL)
	{
		printf("There is a problem with the dynamic memory \n");
		FreeMember(&member);
		exit(1);
	}
	strcpy(member.password, password);//insert the password into the password field, in newmember struct
	printf("Please enter a description about yourself, maximum 200 chars\n");
	gets(desc);
	member.decription = (char*)malloc((strlen(desc) + 1) * sizeof(char));//get the exactly fields to insert description 
	if (member.decription == NULL)
	{
		printf("There is a problem with the dynamic memory \n");
		FreeMember(&member);
		exit(1);
	}
	strcpy(member.decription, desc);//insert the descritpion into the description field, in newmember struct
	member.hobbies = (char**)malloc((NumOfHobbies) * sizeof(char*));//create dynamic array ** for 4 hobbies
	if (member.hobbies == NULL)
	{
		printf("There is a problem with the dynamic memory \n");
		FreeMember(&member);
		exit(1);
	}

	for (i = 0; i < NumOfHobbies; i++)
	{
		printf("Please choose the %d hobbie from the list : TV_Shows, Movies, Gym, Basketball,\nBaseball, Bicycle, Poetry, Books, Drawing ,Theatre . \n", i + 1);
		gets(hobbie);
		member.hobbies[i] = (char*)malloc((strlen(hobbie) + 1) * sizeof(char));//create the dynamic array for 1hobbie with the exactly fields
		if (member.hobbies[i] == NULL)
		{
			printf("There is a problem with the dynamic memory \n");
			FreeMember(&member);
			exit(1);
		}
		strcpy(member.hobbies[i], hobbie);//insert the hobbie 'i' into the hobbie[i] field, in the hobbie array
	}
	*size = *size + 1;//we must up the size by 1, cuz we have another member to the community array
	community->community = (Newmember*)realloc(community->community, (*size) * sizeof(Newmember)); //we need to do realloc to add another field for the new member
	community->community[(*size) - 1] = member; //insert the newmember to the right place in the field, size-1 means the last slot in the field
}

void FindMatch(community community, int index, int size)
{
	int fromage, toage, Hmatch = 0, matches = 0;
	int i, j, k;
	printf("enter age range\n");
	scanf("%d %d", &fromage, &toage);

	for (i = 0; i < size; i++)
	{
		if (community.community[index].gender != community.community[i].gender)//if the gender of the user is not equal to the match
			if (community.community[i].age < toage && community.community[i].age >= fromage)//if the age of the match is between the range of the age that the user entered
				for (j = 0; j < NumOfHobbies; j++)//for  loop to  compare every user's hobbie to match hobbie
					for (k = 0; k < NumOfHobbies; k++)//check every match's hobbie
					{
						if (strcmp(community.community[index].hobbies[j], community.community[i].hobbies[k]) == Success)//if the user's hobbie equal to the match's hobbie
							Hmatch++;//we add 1 cuz we found 1 match 
					}
		if (Hmatch >= 2)//if after the loops, found that there is a match for at least two hobbies between user and the match
		{
			matches++;
			PrintMatch(community, i);//we can print the match
			printf("\n\n");//seperate between every match with two lines
		}
		Hmatch = 0;//after we check 1 match, we have to check another matches, so the hobbie count is must be 0
	}
	if (matches == 0)
	{
		printf("No match found! Sorry...\n");
	}
	MainMenuOptions(community, index, size);
}

void PrintMatch(community community, int index)
{



	printf("Name:%s %s\n", community.community[index].fname, community.community[index].lname);//print the first name and last of the match
	printf("Age:%d\n", community.community[index].age);//print the match's age
	printf("Description:%s\n", community.community[index].decription);//print the match's description
	printf("Hobbies:");
	printf("%s, %s, %s, %s.", community.community[index].hobbies[hobbie1], community.community[index].hobbies[hobbie2], community.community[index].hobbies[hobbie3], community.community[index].hobbies[hobbie4]);//print the match's hobbies


}
void ReadInformation(community* community, int* size, int* index)
{
	char end = '0', help;
	int i, sendsize;
	char buffer[DescSIZE], temp[DescSIZE];
	Newmember member;
	FILE* file;
	file = fopen("input.txt", "a+");// we read the text file firstly to get all the information with a+ , so the pointer will start to read before EOF
	if (file == NULL)
	{
		printf("There is a problem with opening the file \n");
		exit(1);
	}
	fseek(file, 0, SEEK_SET);// we want to read from the first place in the text, so we must make the pointer's position at the beginning
	while (end != EOF)//we can get every information until we reached EOF
	{

		end = fscanf(file, "%s", buffer);//we get the id to the buffer(varaible)

		if (end != EOF)// if we didn't read EOF
		{
			member.id = (char*)malloc((ID + 1) * sizeof(char));//get a dynamic array to the id field in the member struct
			if (member.id == NULL)
			{
				printf("There is a problem with the dynamic memory \n");
				FreeMember(&member);
				exit(1);
			}
			strcpy(member.id, buffer);//we insert the information in the buffer(id) to the id field in the struct
			end = fscanf(file, "%s", buffer);//we get the firstname to the buffer(varaible)
			member.fname = (char*)malloc((strlen(buffer) + 1) * sizeof(char));//get a dynamic array to the firstname field in the member struct
			if (member.fname == NULL)
			{
				printf("There is a problem with the dynamic memory \n");
				FreeMember(&member);
				exit(1);
			}
			strcpy(member.fname, buffer);//we insert the information in the buffer(firstname) to the firstname field in the struct
			end = fscanf(file, "%s", buffer);//we get the lastname to the buffer(varaible)
			member.lname = (char*)malloc((strlen(buffer) + 1) * sizeof(char));//get a dynamic array to the lastname field in the member struct
			if (member.lname == NULL)
			{
				printf("There is a problem with the dynamic memory \n");
				FreeMember(&member);
				exit(1);
			}
			strcpy(member.lname, buffer);//we insert the information in the buffer(lastname) to the lastname field in the struct
			end = fscanf(file, "%d", &member.age);//we insert the age to the age field in the struct
			end = fscanf(file, "%c", &help);//there's a space there
			end = fscanf(file, "%c", &member.gender);//we do it again to get the gender,we insert the gender to the gender field in the struct
			end = fscanf(file, "%s", buffer);//we get the username to the buffer(varaible)
			member.username = (char*)malloc((strlen(buffer) + 1) * sizeof(char));//get a dynamic array to the username field in the member struct
			if (member.username == NULL)
			{
				printf("There is a problem with the dynamic memory \n");
				FreeMember(&member);
				exit(1);
			}
			strcpy(member.username, buffer);//we insert the information in the buffer(username) to the username field in the struct
			end = fscanf(file, "%s", buffer);//we get the password to the buffer(varaible)
			member.password = (char*)malloc((strlen(buffer) + 1) * sizeof(char));//get a dynamic array to the password field in the member struct
			if (member.password == NULL)
			{
				printf("There is a problem with the dynamic memory \n");
				FreeMember(&member);
				exit(1);
			}
			strcpy(member.password, buffer);//we insert the information in the buffer(password) to the password field in the struct
			end = fscanf(file, "%s", temp);//we get the descripition's first word to the temp(varaible)
			while (strcmp(buffer, ";") != 0)// when we read from the text file, we read all words until we found delimeter
			{
				strcat(temp, " ");//we concatenate ' ' to the temp
				end = fscanf(file, "%s", buffer);//we read the next word
				if (strcmp(buffer, ";") == 0)// if the word we read is ; so we stop
					break;
				else
					strcat(temp, buffer);//we cocatenate to temp the word in the buffer
			}
			member.decription = (char*)malloc((strlen(temp) + 1) * sizeof(char));//get a dynamic array to the description field in the member struct
			if (member.decription == NULL)
			{
				printf("There is a problem with the dynamic memory \n");
				FreeMember(&member);
				exit(1);
			}
			strcpy(member.decription, temp);//we insert the information in the temp(description) to the description field in the struct

			member.hobbies = (char**)malloc((NumOfHobbies) * sizeof(char*));//we get a dyanmic memory to hobbie array of 4 places
			if (member.hobbies == NULL)
			{
				printf("There is a problem with the dynamic memory \n");
				FreeMember(&member);
				exit(1);
			}
			for (i = 0; i < NumOfHobbies; i++)
			{
				end = fscanf(file, "%s", buffer);//we get the hobbie[i] to the buffer(varaible)
				member.hobbies[i] = (char*)malloc((strlen(buffer) + 1) * sizeof(char));//get a dynamic array to the hobbie[i] field in the hobbie array
				if (member.hobbies[i] == NULL)
				{
					printf("There is a problem with the dynamic memory \n");
					FreeMember(&member);
					exit(1);
				}
				strcpy(member.hobbies[i], buffer);//we insert the information in the buffer(hobbie) to the hobbie[i] field in the hobbie array
			}
			if (*size > 1)// if size is bigger than 1, so we must doing realloc, if it was 1 the malloc that we did in EnterMenuOption() is enough
				community->community = (Newmember*)realloc(community->community, (*size) * sizeof(Newmember));
			if (community->community == NULL)
			{
				printf("There is a problem with the dynamic memory \n");
				sendsize = *size;
				FreeMemory(community->community, sendsize);
				exit(1);
			}
			community->community[*index] = member;// we insert every member in community[index]
			*index = *index + 1;//we must add 1 to the index for the next place
			*size = *size + 1;// we must add 1 to the size, so we can add another members if needed
		}
	}
	*size = *size - 1;// cuz we started with 1 in the EnterMenuOption(), we must decrease 1. we must start with 1 in StartMenu, cuz we can't do malloc to size if size 0.
	fclose(file);//after we finished reading the information, we close the text file.
}
void FreeMemory(community* community, int size)
{
	int i, j;
	for (i = 0; i < size; i++)// for loop to free the hobbies of all memebers firstly
	{
		for (j = 0; j < NumOfHobbies; j++)
		{
			free(community->community[i].hobbies[j]);//free the community[i] 's hobbie[i]
		}
	}
	for (i = 0; i < size; i++)// we free every member in the community[i]
	{
		free(community->community[i].decription);//free the community[i] 's description
		free(community->community[i].password);//free the community[i] 's password
		free(community->community[i].username);//free the community[i] 's username
		free(community->community[i].lname);//free the community[i] 's lastname
		free(community->community[i].fname);//free the community[i] 's firstname
		free(community->community[i].hobbies);//free the community[i] 's hobbie array
		free(community->community[i].id);//free the community[i] 's id
	}
	free(community->community);//free the community struct
}
void FreeMember(Newmember* member)
{
	int i;
	free(member->id);//free the member's id
	free(member->fname);//free the member's fname
	free(member->lname);//free the member's lastname
	free(member->username);//free the member's username
	free(member->password);//free the member's password
	free(member->decription);//free the member's description
	for (i = 0; i < NumOfHobbies; i++)
		free(member->hobbies[i]);//free the member's hobbie[i]
	free(member->hobbies);//free the hobbie array in member
	free(member);//free the member struct
}