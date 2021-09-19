// injector.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "windows.h"
#include <string>
#include <regex>
using namespace std;
void Inject(DWORD pId, const WCHAR * dll_path);



int main()
{
	string buffer;
	DWORD pid;
	cout << "Enter the pid, format is Injector<PID>:";
	cin >> buffer;
	buffer=regex_replace(buffer, std::regex("[^0-9]+"),"");
	pid = stoi(buffer);
	const WCHAR dll_path[] = L"C:\\Users\\orell\\Downloads\\injected.dll";
	Inject(pid, dll_path);
	return 0;
}

void Inject(DWORD pId, const WCHAR * dll_path)
{
	HANDLE h = OpenProcess(PROCESS_ALL_ACCESS, false, pId);
	if (h)
	{
		FARPROC  LoadLibAddr = GetProcAddress(GetModuleHandleA("kernel32.dll"), "LoadLibraryW");
		const int dll_path_size = (wcslen(dll_path) + 1) * sizeof(WCHAR); // or sizeof(dll_path);
		LPVOID dllName_alloc = VirtualAllocEx(h, NULL, dll_path_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
		WriteProcessMemory(h, dllName_alloc, dll_path, dll_path_size, NULL);
		HANDLE r_t_hdle = CreateRemoteThread(h, NULL, NULL, reinterpret_cast<LPTHREAD_START_ROUTINE>(LoadLibAddr), dllName_alloc, 0, NULL);
		WaitForSingleObject(r_t_hdle, INFINITE);
		DWORD exitCode = 0;
		GetExitCodeThread(r_t_hdle, &exitCode);
		if (exitCode != 0)
			cout << "DLL loaded successfully!" << endl;
		else
			cout << "DLL load failed!" << endl;
		VirtualFreeEx(h, dllName_alloc, dll_path_size, MEM_RELEASE);
		CloseHandle(r_t_hdle);
		CloseHandle(h);
		
	}
	else
		cout << "OpenProcess failed!" << endl;
}
