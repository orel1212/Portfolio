// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include <windows.h>
#include<iostream>
using namespace std;
BOOL CALLBACK EnumWindowsProc(HWND hWnd, LPARAM lParam);
BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
					 )
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		EnumWindows(EnumWindowsProc, NULL);
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}
BOOL CALLBACK EnumWindowsProc(HWND hWnd, LPARAM lParam) {
	DWORD process;
	//WCHAR title[80];
	GetWindowThreadProcessId(hWnd, &process);
	OutputDebugStringA("Thread Proccess ID:" + process);
	OutputDebugStringA("Current Proccess ID:"+GetCurrentProcessId());
	if (GetCurrentProcessId() == process) {
		//GetWindowText(hWnd, title, sizeof(title));
		WCHAR str_pwn[] = L"NOTEPAD is PWNED by senia & orel!";
		SetWindowText(hWnd, str_pwn);
		return FALSE;
	}
	return TRUE;
}
