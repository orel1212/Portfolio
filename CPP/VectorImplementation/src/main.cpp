#include <iostream>
#include <string>
#include "Vector.h"


int main()
{
	Vector<int> my_vector_int;
	std::cout << "Vector<int>:\n";
	my_vector_int.PushBack(1);
	my_vector_int.PushBack(2);
	std::cout << my_vector_int;
	my_vector_int.PopBack();
	my_vector_int.PopBack();
	std::cout << my_vector_int;
	my_vector_int.PushBack(2);
	my_vector_int.PushBack(3);
	std::cout << my_vector_int;
	my_vector_int.PopBack();
	std::cout << my_vector_int;
	my_vector_int.PopBack();
	std::cout << my_vector_int;

	std::cout << "Vector<string>:\n";
	Vector<std::string> my_vector_str;
	my_vector_str.EmplaceBack("Orel");
	std::cout << my_vector_str;
	my_vector_str.PopBack();
	std::cout << my_vector_str;
	my_vector_str.EmplaceBack("Orel");
	my_vector_str.EmplaceBack("Lavie");
	std::cout << my_vector_str;
	my_vector_str.PopBack();
	std::cout << my_vector_str;
	my_vector_str.PopBack();
	std::cout << my_vector_str;

	std::cin.get();
	return 0;
}