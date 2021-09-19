
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct
{
	int first;
	int second;
} Pair;
void isReflexive(int* groupA, int size, Pair* pairs, int numofpairs, int* flag);
void isIrreflexive(int* groupA, int size, Pair* pairs, int numofpairs);
void isSymmetric(Pair* pairs, int numofpairs, int* flag);
void isTransitive(Pair* pairs, int size, int numofpairs, int* flag);
void isAntiSymmetric(Pair* pairs, int numofpairs);
void isASymmetric(Pair* pairs, int numofpairs);
void isEquivalence(Pair* pairs, int numofpairs, int size, int reflexiveflag, int symmetryflag, int transitiveflag);
void MatrixR(Pair* pairs, int size, int numofpairs);
void MatrixRPowTwo(Pair* pairs, int size, int numofpairs);
void MatrixRPowThree(Pair* pairs, int size, int numofpairs);
void MatrixRInfiniteAndRAsterisk(Pair* pairs, int size, int numofpairs);
int main()
{
	FILE* file;//a pointer to the file
	file = fopen("r.txt", "r");
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int reflexiveflag = 1, symmetryflag = 1, transitiveflag = 1;
	char end = '0';
	int size;
	int index = 0;
	int numofpairs = 0;
	Pair* pairs;
	fseek(file, 0, SEEK_SET);
	fscanf(file, "%d", &size);
	int* groupA = (int*)malloc((size) * sizeof(int));
	if (groupA == NULL)
	{
		printf("There is a problem with the dynamic memory \n");
		exit(1);
	}
	fscanf(file, "%*[^\n]\n", NULL);
	while (end != EOF)//we can get every information until we reached EOF
	{
		end = fscanf(file, "%d", &index);
		numofpairs++;
	}
	numofpairs /= 2;
	pairs = (Pair*)malloc((numofpairs) * sizeof(Pair));
	if (pairs == NULL)
	{
		printf("There is a problem with the dynamic memory \n");
		exit(1);
	}
	for (index = 0; index < size; index++)
	{
		groupA[index] = index + 1;
	}
	end = '0';
	index = 0;
	fseek(file, 0, SEEK_SET);
	fscanf(file, "%*[^\n]\n", NULL);
	while (end != EOF)//we can get every information until we reached EOF
	{
		end = fscanf(file, "%d", &(pairs[index].first));
		end = fscanf(file, "%d", &(pairs[index].second));
		index++;
	}
	fclose(file);//after we finished reading the information, we close the text file.
	isReflexive(groupA, size, pairs, numofpairs, &reflexiveflag);
	isIrreflexive(groupA, size, pairs, numofpairs);
	isSymmetric(pairs, numofpairs, &symmetryflag);
	isAntiSymmetric(pairs, numofpairs);
	isASymmetric(pairs, numofpairs);
	isTransitive(pairs, size, numofpairs, &transitiveflag);
	isEquivalence(pairs, numofpairs, size, reflexiveflag, symmetryflag, transitiveflag);
	MatrixR(pairs, size, numofpairs);
	MatrixRPowTwo(pairs, size, numofpairs);
	MatrixRPowThree(pairs, size, numofpairs);
	MatrixRInfiniteAndRAsterisk(pairs, size, numofpairs);
	free(pairs);
	free(groupA);
}
void isReflexive(int* groupA, int size, Pair* pairs, int numofpairs, int* flag)
{
	FILE* file;//a pointer to the file
	file = fopen("output.txt", "w");
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int i, j;
	Pair temp;
	int flagpair;
	for (i = 0; i < size; i++)
	{
		flagpair = 0;
		temp.first = groupA[i];
		temp.second = groupA[i];
		for (j = 0; j < numofpairs; j++)
		{
			if (temp.first == pairs[j].first && temp.second == pairs[j].second)
			{
				flagpair = 1;
				break;
			}
		}
		if (flagpair == 0)
		{
			fputs("R is not Reflexsive\n", file);
			*(flag) = 0;
			break;
		}
	}
	if (flagpair == 1)
		fputs("R is Reflexsive\n", file);
	fclose(file);
}
void isIrreflexive(int* groupA, int size, Pair* pairs, int numofpairs)
{
	FILE* file;//a pointer to the file
	file = fopen("output.txt", "a+");
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int i, j;
	Pair temp;
	int flagpair = 0;
	for (i = 0; i < size; i++)
	{
		temp.first = groupA[i];
		temp.second = groupA[i];
		for (j = 0; j < numofpairs; j++)
		{
			if (temp.first == pairs[j].first && temp.second == pairs[j].second)
			{
				flagpair = 1;
				break;
			}
		}
		if (flagpair == 1)
		{
			fputs("R is not irreflexsive\n", file);
			break;
		}
	}
	if (flagpair == 0)
		fputs("R is irreflexsive\n", file);
	fclose(file);
}
void isSymmetric(Pair* pairs, int numofpairs, int* flag)
{
	FILE* file;//a pointer to the file
	file = fopen("output.txt", "a+");// we read the text file firstly to get all the information with a+ , so the pointer will start to read before EOF
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int i, j;
	int flagpair;
	for (i = 0; i < numofpairs; i++)
	{
		if (pairs[i].first != pairs[i].second)
		{
			flagpair = 0;
			for (j = 0; j < numofpairs; j++)
			{
				if (i != j)
				{
					if (pairs[i].first == pairs[j].second && pairs[i].second == pairs[j].first)
					{
						flagpair = 1;
						break;
					}
				}
			}
			if (flagpair == 0)
			{
				fputs("R is not Symmetric\n", file);
				*(flag) = 0;
				break;
			}
		}
	}
	if (flagpair == 1)
		fputs("R is Symmetric\n", file);
	fclose(file);
}
void isAntiSymmetric(Pair* pairs, int numofpairs)
{
	FILE* file;//a pointer to the file
	file = fopen("output.txt", "a+");// we read the text file firstly to get all the information with a+ , so the pointer will start to read before EOF
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int i, j;
	int flagpair = 0;
	for (i = 0; i < numofpairs; i++)
	{
		for (j = 0; j < numofpairs; j++)
		{
			if (i != j)
			{
				if (pairs[i].first == pairs[j].second && pairs[i].second == pairs[j].first)
				{
					if (pairs[i].first != pairs[i].second)
					{
						flagpair = 1;
						break;
					}
				}
			}
		}
		if (flagpair == 1)
		{
			fputs("R is not AntiSymmetric\n", file);
			break;
		}
	}
	if (flagpair == 0)
		fputs("R is AntiSymmetric\n", file);
	fclose(file);
}
void isASymmetric(Pair* pairs, int numofpairs)
{
	FILE* file;//a pointer to the file
	file = fopen("output.txt", "a+");// we read the text file firstly to get all the information with a+ , so the pointer will start to read before EOF
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int i, j;
	int flagpair = 0;
	for (i = 0; i < numofpairs; i++)
	{
		for (j = 0; j < numofpairs; j++)
		{
			if (i != j)
			{
				if (pairs[i].first == pairs[j].second && pairs[i].second == pairs[j].first)
				{
					flagpair = 1;
					break;
				}
			}
		}
		if (flagpair == 1)
		{
			fputs("R is not ASymmetric\n", file);
			break;
		}
	}
	if (flagpair == 0)
		fputs("R is ASymmetric\n", file);
	fclose(file);
}
void isTransitive(Pair* pairs, int size, int numofpairs, int* flag)
{
	FILE* file;//a pointer to the file
	file = fopen("output.txt", "a+");// we read the text file firstly to get all the information with a+ , so the pointer will start to read before EOF
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int i, j, k;
	int indexpair = 0;
	int** matrix = (int**)malloc((size) * sizeof(int*));
	if (matrix == NULL)
	{
		printf("Dynamic memory failed \n");
		exit(1);
	}
	for (i = 0; i < size; i++)
	{
		matrix[i] = (int*)malloc((size) * sizeof(int));
		if (matrix[i] == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		for (j = 0; j < size; j++)
		{
			matrix[i][j] = 0;
		}
	}
	while (indexpair < numofpairs)
	{
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				if (i + 1 == pairs[indexpair].first && j + 1 == pairs[indexpair].second)
				{
					matrix[i][j] = 1;
					indexpair++;
				}
			}
		}
	}
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			for (k = 0; k < size; k++)
			{
				if (matrix[i][k] == 1 && matrix[k][j] == 1 && matrix[i][j] == 0)
				{
					fputs("R is not Transitive\n", file);
					fclose(file);
					*(flag) = 0;
					for (i = 0; i < size; i++)
						free(matrix[i]);
					free(matrix);
					return;
				}
			}
		}
	}
	fputs("R is Transitive\n", file);
	for (i = 0; i < size; i++)
		free(matrix[i]);
	free(matrix);
	fclose(file);
}
void isEquivalence(Pair* pairs, int numofpairs, int size, int reflexiveflag, int symmetryflag, int transitiveflag)
{
	FILE* file;//a pointer to the file
	file = fopen("output.txt", "a+");// we read the text file firstly to get all the information with a+ , so the pointer will start to read before EOF
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int i, j, flag = 0;
	if (reflexiveflag == 0 || symmetryflag == 0 || transitiveflag == 0)
		fputs("R is not Equivalence\n", file);
	else
	{
		int indexpair = 0;
		int** matrix = (int**)malloc((size) * sizeof(int*));
		if (matrix == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		int* arr = (int*)malloc((size) * sizeof(int));
		if (arr == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		for (i = 0; i < size; i++)
		{
			if (i != 0)
				arr[i] = 0;
			else
				arr[i] = 1;
			matrix[i] = (int*)malloc((size) * sizeof(int));
			if (matrix[i] == NULL)
			{
				printf("Dynamic memory failed \n");
				exit(1);
			}
			for (j = 0; j < size; j++)
			{
				matrix[i][j] = 0;
			}
		}
		while (indexpair < numofpairs)
		{
			for (i = 0; i < size; i++)
			{
				for (j = 0; j < size; j++)
				{
					if (i + 1 == pairs[indexpair].first && j + 1 == pairs[indexpair].second)
					{
						matrix[i][j] = 1;
						indexpair++;
					}
				}
			}
		}
		fputs("R is Equivalence\n", file);
		fputs("Equivalence classes:\n", file);

		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				if (matrix[i][j] == 1)
				{
					if (arr[j] == 0)
						arr[j] = i + 1;
				}
			}
		}
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				if (arr[j] == i + 1)
				{
					fprintf(file, "%d", j + 1);
					fprintf(file, "%c", ' ');
					flag = 1;
				}
			}
			if (flag == 1)
			{
				fputs("\n", file);
				flag = 0;
			}
		}
		for (i = 0; i < size; i++)
			free(matrix[i]);
		free(matrix);
		free(arr);
	}
	fclose(file);
}
void MatrixR(Pair* pairs, int size, int numofpairs)
{
	FILE* file;//a pointer to the file
	file = fopen("output.txt", "a+");// we read the text file firstly to get all the information with a+ , so the pointer will start to read before EOF
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int i, j;
	int indexpair = 0;
	int** matrix = (int**)malloc((size) * sizeof(int*));
	if (matrix == NULL)
	{
		printf("Dynamic memory failed \n");
		exit(1);
	}
	for (i = 0; i < size; i++)
	{
		matrix[i] = (int*)malloc((size) * sizeof(int));
		if (matrix[i] == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		for (j = 0; j < size; j++)
		{
			matrix[i][j] = 0;
		}
	}
	while (indexpair < numofpairs)
	{
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				if (i + 1 == pairs[indexpair].first && j + 1 == pairs[indexpair].second)
				{
					matrix[i][j] = 1;
					indexpair++;
				}
			}
		}
	}
	fputs("Maxtrix of R:\n", file);
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			fprintf(file, "%d ", matrix[i][j]);
		}
		fputs("\n", file);
		free(matrix[i]);
	}
	free(matrix);
	fclose(file);
}
void MatrixRPowTwo(Pair* pairs, int size, int numofpairs)
{
	FILE* file;//a pointer to the file
	file = fopen("output.txt", "a+");// we read the text file firstly to get all the information with a+ , so the pointer will start to read before EOF
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int i, j, k;
	int indexpair = 0;
	int** matrix2;
	int** matrix = (int**)malloc((size) * sizeof(int*));
	if (matrix == NULL)
	{
		printf("Dynamic memory failed \n");
		exit(1);
	}
	for (i = 0; i < size; i++)
	{
		matrix[i] = (int*)malloc((size) * sizeof(int));
		if (matrix[i] == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		for (j = 0; j < size; j++)
		{
			matrix[i][j] = 0;
		}
	}
	while (indexpair < numofpairs)
	{
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				if (i + 1 == pairs[indexpair].first && j + 1 == pairs[indexpair].second)
				{
					matrix[i][j] = 1;
					indexpair++;
				}
			}
		}
	}
	matrix2 = (int**)malloc((size) * sizeof(int*));
	if (matrix2 == NULL)
	{
		printf("Dynamic memory failed \n");
		exit(1);
	}
	for (i = 0; i < size; i++)
	{
		matrix2[i] = (int*)malloc((size) * sizeof(int));
		if (matrix2[i] == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		for (j = 0; j < size; j++)
		{
			matrix2[i][j] = 0;
		}
	}
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			for (k = 0; k < size; k++)
			{
				if (matrix[i][k] == 1 && matrix[k][j] == 1)
				{
					matrix2[i][j] = 1;
				}
			}
		}
	}
	fputs("Maxtrix of R^2:\n", file);
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			fprintf(file, "%d ", matrix2[i][j]);
		}
		fputs("\n", file);
		free(matrix[i]);
		free(matrix2[i]);
	}
	free(matrix);
	free(matrix2);
	fclose(file);
}
void MatrixRPowThree(Pair* pairs, int size, int numofpairs)
{
	FILE* file;//a pointer to the file
	file = fopen("output.txt", "a+");// we read the text file firstly to get all the information with a+ , so the pointer will start to read before EOF
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int i, j, k, p;
	int indexpair = 0;
	int** matrix2;
	int** matrix = (int**)malloc((size) * sizeof(int*));
	if (matrix == NULL)
	{
		printf("Dynamic memory failed \n");
		exit(1);
	}
	for (i = 0; i < size; i++)
	{
		matrix[i] = (int*)malloc((size) * sizeof(int));
		if (matrix[i] == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		for (j = 0; j < size; j++)
		{
			matrix[i][j] = 0;
		}
	}
	while (indexpair < numofpairs)
	{
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				if (i + 1 == pairs[indexpair].first && j + 1 == pairs[indexpair].second)
				{
					matrix[i][j] = 1;
					indexpair++;
				}
			}
		}
	}
	matrix2 = (int**)malloc((size) * sizeof(int*));
	if (matrix2 == NULL)
	{
		printf("Dynamic memory failed \n");
		exit(1);
	}
	for (i = 0; i < size; i++)
	{
		matrix2[i] = (int*)malloc((size) * sizeof(int));
		if (matrix2[i] == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		for (j = 0; j < size; j++)
		{
			matrix2[i][j] = 0;
		}
	}
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			for (k = 0; k < size; k++)
			{
				for (p = 0; p < size; p++)
				{
					if (matrix[i][k] == 1 && matrix[k][j] == 1 && matrix[j][p] == 1)
					{
						matrix2[i][p] = 1;
					}
				}
			}
		}
	}
	fputs("Maxtrix of R^3:\n", file);
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			fprintf(file, "%d ", matrix2[i][j]);
		}
		fputs("\n", file);
		free(matrix[i]);
		free(matrix2[i]);
	}
	free(matrix);
	free(matrix2);
	fclose(file);
}
void MatrixRInfiniteAndRAsterisk(Pair* pairs, int size, int numofpairs)
{
	FILE* file;//a pointer to the file
	file = fopen("output.txt", "a+");// we read the text file firstly to get all the information with a+ , so the pointer will start to read before EOF
	if (file == NULL)
	{
		printf("There is a problem with openning the file \n");
		exit(1);
	}
	int i, j, k, p, sum = 0;
	int indexpair = 0;
	int** matrix = (int**)malloc((size) * sizeof(int*));
	if (matrix == NULL)
	{
		printf("Dynamic memory failed \n");
		exit(1);
	}
	int** matrix2 = (int**)malloc((size) * sizeof(int*));
	if (matrix2 == NULL)
	{
		printf("Dynamic memory failed \n");
		exit(1);
	}
	int** matrix3 = (int**)malloc((size) * sizeof(int*));
	if (matrix3 == NULL)
	{
		printf("Dynamic memory failed \n");
		exit(1);
	}
	int** matrix4 = (int**)malloc((size) * sizeof(int*));
	if (matrix3 == NULL)
	{
		printf("Dynamic memory failed \n");
		exit(1);
	}
	for (i = 0; i < size; i++)
	{
		matrix[i] = (int*)malloc((size) * sizeof(int));
		if (matrix[i] == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		matrix2[i] = (int*)malloc((size) * sizeof(int));
		if (matrix2[i] == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		matrix3[i] = (int*)malloc((size) * sizeof(int));
		if (matrix3[i] == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		matrix4[i] = (int*)malloc((size) * sizeof(int));
		if (matrix3[i] == NULL)
		{
			printf("Dynamic memory failed \n");
			exit(1);
		}
		for (j = 0; j < size; j++)
		{
			matrix[i][j] = 0;
			matrix2[i][j] = 0;
			matrix3[i][j] = 0;
			matrix4[i][j] = 0;
		}
	}
	while (indexpair < numofpairs)
	{
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				if (i + 1 == pairs[indexpair].first && j + 1 == pairs[indexpair].second)
				{
					matrix[i][j] = 1;
					matrix2[i][j] = 1;
					matrix3[i][j] = 1;
					indexpair++;
				}
			}
		}
	}
	for (p = 0; p < size - 1; p++)
	{
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				for (k = 0; k < size; k++)
				{
					sum = sum + matrix2[i][k] * matrix3[k][j];
				}
				if (sum > 1)
					sum = 1;
				matrix4[i][j] = sum;
				sum = 0;
			}
		}
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				matrix2[i][j] = matrix4[i][j];
			}
		}
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				matrix[i][j] = matrix[i][j] + matrix2[i][j];
				if (matrix[i][j] > 1)
					matrix[i][j] = 1;
			}
		}
	}
	fputs("Maxtrix of R^Inf:\n", file);
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			fprintf(file, "%d ", matrix[i][j]);
		}
		fputs("\n", file);
	}
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			if (i == j)
				matrix[i][j] = 1;
		}
	}
	fputs("Maxtrix of R*:\n", file);
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			fprintf(file, "%d ", matrix[i][j]);
		}
		fputs("\n", file);
		free(matrix[i]);
		free(matrix2[i]);
		free(matrix3[i]);
		free(matrix4[i]);
	}
	free(matrix);
	free(matrix2);
	free(matrix3);
	free(matrix4);
	fclose(file);
}
