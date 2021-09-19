
#ifndef STRING_H
#define STRING_H
#define _CRT_SECURE_NO_WARNINGS
#include<iostream>
#include<fstream>
#include <assert.h>
#define MaxSizeD 201
#define MaxSizeW 81
#define NoSize 0
#define NotPossibleINDEX 0
typedef enum {Smaller=-1,Equal,Bigger} CMP;
using namespace std;
class String
{
	char * str;
	int size;
public:
	String(const char* str = NULL);//c'tor that get dynamic array of chars
	String(const String& str);//copy c'tor that gets string obj
	bool operator==(const String&) const;//operator == that gets string obj,return true if they are similiar
	CMP operator<(const String&) const;//operator < that get string obj and check if the this obj smallest than the the string obj,return CMP type
	String& operator=(const String&);//operator = that gets string obj
	String& operator+=(const char&);//operator += that get char and add to the string
	bool operator!=(const String& str) const;//operator != that gets string obj
	friend ostream& operator<<(ostream& out, const String& str);//ostream operator that get ostream obj and string obj
	friend istream& operator>>(istream& in, String& str);//istream operator that get istream obj and string obj
	String& operator-=(const char& remove);//operator -= that get char and remove from the string
	char& operator[](const int&) const;//operator [] that get index and return the char in this index
	inline int& strlen()//strlen function that return the length of the string
	{
		return this->size;
	}
	inline int strlen(const String& str)//get string obj and return his size
	{
		return str.size;
	}
	int strlen(const char * str);//return the length of char * array
	~String();//d'tor
	void Repair();//repair the word&defenition
};
#endif