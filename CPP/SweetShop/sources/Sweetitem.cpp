
#include"Sweetitem.h"
int Sweetitem::serialnum=0;//make the serial num staring from 0
Sweetitem::Sweetitem()
{
}
void Sweetitem::AddOneToSNUM()
{
	serialnum++;//add one if there is new sweetitem
}
void Sweetitem::ReduceOneFromSNUM()
{
	serialnum--;//reduce one is one sweet item is removed
}
Sweetitem::~Sweetitem()
{
}
