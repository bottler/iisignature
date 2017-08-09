#pragma once


#define WIN32_LEAN_AND_MEAN
#include<Windows.h>
//#define PSAPI_VERSION 1
#include <psapi.h>
#include<iostream>

//arg could e.g. be __LINE__
void printWorkingSetSize(int arg) {
  PROCESS_MEMORY_COUNTERS pmc;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    std::cerr << "failed to get process memory info\n";
  std::cout << arg << " Working set: " << pmc.WorkingSetSize / 1000000 << 
    "Mb  . Peak:" << pmc.PeakWorkingSetSize/1000000 << "Mb.\n";
}

void restrictWorkingSet(size_t mb){
  SetProcessWorkingSetSizeEx(GetCurrentProcess(), 204800, mb * 1024 * 1024, 
    QUOTA_LIMITS_HARDWS_MAX_ENABLE|QUOTA_LIMITS_HARDWS_MIN_DISABLE);
}

