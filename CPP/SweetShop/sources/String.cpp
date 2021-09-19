
#include "String.h"
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// constructor/default constructor
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
String::String(const char* str)
{
	if (str)
	{
		this->size = strlen(str);
		this->str = new char[this->size+1];
		this->str[this->size] = '\0';//add \0 to the end of the str
		for (int i = 0; str[i] != '\0'; i++)//copy
			this->str[i] = str[i];

	}
	else
	{
		this->str = NULL;//str is empty
		this->size = 0;//size =0
	}
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// copy c'tor
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
String::String(const String& str)
{
	this->size = strlen(str);
	if (str.str)//if the str isn't empty
	{
		this->str = new char[this->size + 1];
		for (int i = 0; str.str[i] != '\0'; i++)//copy
			this->str[i] = str.str[i];
		this->str[this->size] = '\0';//add \0 to the end of the str
	}
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// destructor
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
String::~String()
{
	if (this->str)//if the str isn't empty
	{
		delete[] this->str;//free memory
		this->str = NULL;
	}
}
int String::strlen(const char * str)
{
	int count = 0;
	while (str[count] != '\0')//while loop that check what the length of the str
	{
		count++;
	}
	return count;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// operator =
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
String& String::operator = (const String& str)
{
	//checking self-assignments
	if (this != &str)//if the obj isn't the same obj
	{
		//check if there's memory to release
		if (this->str)
		{
			delete[] this->str;
			this->str = NULL;
		}
		//copy str to currect string
		this->size = strlen(str);
		if (str.str)
		{
			this->str = new char[this->size + 1];
			for (int i = 0; str.str[i] != '\0'; i++)
				this->str[i] = str.str[i];
			this->str[this->size] = '\0';
		}
	}

	return *this;
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// operator +=
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
String& String::operator += (const char& add)
{
	
	if (this->str)//if the str isn't empty
	{
		int iNewLen = this->size + 1;
		char* sNewStr = new char[iNewLen + 1];//temp array
		for (int i = 0; this->str[i] != '\0'; i++)//copy from this to temp
			sNewStr[i] = this->str[i];
		sNewStr[iNewLen-1] = add;//add the new char
		sNewStr[iNewLen] = '\0';//add \0
		delete[] this->str;
		this->str = new char[iNewLen + 1];
		//update the member attributes:
		for (int i = 0; sNewStr[i] != '\0'; i++)//copy from temp to this
			this->str[i] = sNewStr[i];
		this->str[iNewLen] = '\0';//add \0
		this->size = iNewLen;
		delete[] sNewStr;//free the temp
	}
	else //str is empty, and just have to get string with size 1
	{
		this->size = 1;
		this->str = new char[this->size + 1];
		this->str[this->size-1] = add;
		this->str[this->size] = '\0';
	}

	return *this;
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// operator -=
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
String& String::operator-=(const char& remove)
{
	if (this->str)//str isn't empty
	{
		int iNewLen = 0;
		for (int i = 0; str[i]!='\0'; i++)//if we not reach \0
		{
			if (this->str[i] != remove)//if we didn't reach the remove char, we count how much the array have to be lower with deleting the char
				iNewLen++;
		}
		char* sNewStr = new char[iNewLen + 1];
		for (int i = 0,j=0; str[i] != '\0'; i++)
		{
			if (this->str[i] != remove)//copy if the chars in the string aren't the remove
			{
				sNewStr[j] = this->str[i];
				j++;
			}
		}
		sNewStr[iNewLen] = '\0';
		delete[] this->str;
		this->str = new char[iNewLen + 1];
		//update the member attributes:
		for (int i = 0; sNewStr[i] != '\0'; i++)
			this->str[i] = sNewStr[i];//copy from temp to this
		this->str[iNewLen] = '\0';
		this->size = iNewLen;
		delete[] sNewStr;//free the temp memory
	}
	return *this;
}
bool String::operator == (const String& str) const
{
	if ((this->str) && (str.str))//if they're both not empty
	{
		for (int i = 0; this->str[i] != '\0'; i++)
		{
			if (this->str[i] != str.str[i])//if only 1 char is not same, so they're not similiar
			{
				return false;
				break;
			}
		}
		return true;
	}
	else
		return ((this->str == NULL) && (str == NULL));//if they're both empty so they both similiar
}
CMP String::operator<(const String& obj) const
{
	int index = 0;
	CMP check = Equal;
	if (this->size <= obj.size)//if the this is smallest/euqal at size than obj
	{
		while (check == Equal && index <= this->size)// until the last char of the this
		{
			if (this->str[index] != obj[index])//if the chars isn't equal
			{
				if (this->str[index] > obj[index])//if the this' char is bigger than obj's char so this bigger
					check = Bigger;
				else
					check = Smaller;//obj bigger
			}
			index++;
		}
	}
	else if (this->size > obj.size)//if the obj is smallest at size than this
	{
		while (check == Equal && index <= obj.size)
		{
			if (this->str[index] != obj[index])//if the chars isn't equal
			{
				if (this->str[index] > obj[index])//if the this' char is bigger than obj's char so this bigger
					check = Bigger;
				else
					check = Smaller;//obj bigger
			}
			index++;
		}
	}
	if (check == Equal)
	{
		if (this->size < obj.size)//if they are equal and the obj is bigger by size, so this smaller
		{
			check = Smaller;
		}
		else
		{
			check = Bigger;//this bigger
		}
	}
	return check;
}
bool String::operator != (const String& str) const
{
	return !((*this) == str);//using ==, if == return false so it means they are not equal
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// friend operator << 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ostream& operator << (ostream& out, const String& str)
{
	//verify that string is not empty to avoid access violation
	if (str.str)
	{
		out << str.str << endl;//out the str obj
	}

	return out;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// friend operator >>
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
istream& operator >> (istream& in, String& str)
{
	char sBuffer[MaxSizeD];//buffer to check what the size of the str would be
	in.getline(sBuffer, MaxSizeD);
	str.size = strlen(sBuffer);
	str = sBuffer;//using = operator
	return in;
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// operator []
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
char& String::operator [] (const int& Index) const
{
	assert(Index >= NotPossibleINDEX && Index<this->size);
	return this->str[Index];//return the char in this index
}
void String::Repair()
{
	int newSize = 1;
	char * temp = new char[this->size + 1];
	if (this->str[0] >= 'a' && this->str[0] <= 'z')//if the first char is not uppercase
		temp[0] = this->str[0] - 'a' + 'A';
	else
		temp[0] = this->str[0];//just have to copy to the temp
	newSize++;
	int i = 1;
	while (this->str[i] != '\0')
	{
		if (this->str[i] != '.')//if the char isn't .
		{
			
			if (this->str[i] >= 'A' && this->str[i] <= 'Z')//if it uppercase we have to change to lowercase
			{
				temp[newSize - 1] = this->str[i] + 'a' - 'A';
				newSize++;
				i++;
			}
			
			else if (this->str[i] == ' ')
			{
				i++;
				while (this->str[i] == ' ')//just have to remain only 1 space,if it not ,.:
					i++;
				if (this->str[i] != ':' && this->str[i] != ',' && this->str[i] != '.' && temp[newSize - 2] != ':' && temp[newSize - 2] != ',' && temp[newSize - 2] != '.')
				{
					temp[newSize - 1] = this->str[i - 1];
					newSize++;
				}
			}
			else if (this->str[i] == ':' || this->str[i] == ',')//if it : or ,
			{
				temp[newSize - 1] = this->str[i];//copy it as it is
				newSize++;
				i++;
				while (this->str[i] == ' ')//just have to remain only 1 space
					i++;
			}
			else //if it lower case or chars like &^%@
			{
				temp[newSize - 1] = this->str[i];
				newSize++;
				i++;
			}
		}
		if (this->str[i] == '.')//if it .
		{
			temp[newSize - 1] = this->str[i];//copy it as it is
			newSize++;
			i++;
			while (this->str[i] == ' ')//have to clean all spaces
				i++;
			if (this->str[i] >= 'a' && this->str[i] <= 'z')//if the first char is lowercase we must change to uppercase
			{
				temp[newSize - 1] = this->str[i] + 'A' - 'a';
				newSize++;
				i++;
			}
			else if (this->str[i] >= 'A' && this->str[i] <= 'Z')//if it uppercase, we copy it as it is
			{
				temp[newSize - 1] = this->str[i];
				newSize++;
				i++;
			}
		}
	}
	delete[] this->str;//clean the first memory
	temp[newSize - 1] = '\0';
	this->str = new char[newSize];
	for (i = 0; temp[i] != '\0'; i++)//create new this and copy from the temp after it changed to be right defenition
		this->str[i] = temp[i];
	delete[] temp;
	this->str[newSize - 1] = '\0';
	this->size = newSize - 1;
}