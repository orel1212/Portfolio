#pragma once
#include "windows.h"
#include<iostream>
using namespace std;
#define IMPORT_TABLE_INDEX 1
bool rewriteThunk(PIMAGE_THUNK_DATA pThunk, void* newFunc);
int IAT_Patching(HMODULE hModule, char * func_name_to_IAT, void * new_func_add, PDWORD func_orginial);


typedef HANDLE(WINAPI *ptrFindFirstFileW)(__in LPCWSTR lpFileName, __out LPWIN32_FIND_DATAW lpFindFileData);
HANDLE WINAPI FindFirstFileW_IAT(__in LPCWSTR lpFileName, __out LPWIN32_FIND_DATAW lpFindFileData);
typedef HANDLE(WINAPI *ptrFindFirstFileExW)(__in LPCWSTR lpFileName, __in FINDEX_INFO_LEVELS fInfoLevelId, __out LPVOID lpFindFileData, __in FINDEX_SEARCH_OPS fSearchOp, __reserved  LPVOID lpSearchFilter, __in DWORD dwAdditionalFlags);
HANDLE WINAPI FindFirstFileExW_IAT(__in LPCWSTR lpFileName, __in FINDEX_INFO_LEVELS fInfoLevelId, __out LPVOID lpFindFileData, __in FINDEX_SEARCH_OPS fSearchOp, __reserved  LPVOID lpSearchFilter, __in DWORD dwAdditionalFlags);
typedef BOOL(WINAPI *ptrFindNextFileW)(__in HANDLE hFindFile, __out  LPWIN32_FIND_DATAW lpFindFileData);
BOOL WINAPI FindNextFileW_IAT(__in HANDLE hFindFile, __out  LPWIN32_FIND_DATAW lpFindFileData);


DWORD originalFindFirstFileW = NULL;
DWORD originalFindFirstFileExW = NULL;
DWORD originalFindNextFileW = NULL;
