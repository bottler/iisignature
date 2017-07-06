import os, sys, numpy as np
from six.moves import range, input

#This demonstrates calling rotinv2dprepare in a limited memory environment so that you can experiment 
#without stalling your session. On windows, the working set is limited and the program can page as much
#as usual. On linux, the address space has a hard limit and will terminate if the limit is exceeded. 

#add the parent directory, so we find our iisignature build if it was built --inplace
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import iisignature

mem_limit_mb=30
mem_limit_b = mem_limit_mb * 1024 * 1024

if os.name=="nt":
	import ctypes
	from ctypes import wintypes

	kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

	def errcheck_bool(result, func, args):
		if not result:
			raise ctypes.WinError(ctypes.get_last_error())
		return args
		
	kernel32.GetCurrentProcess.restype = wintypes.HANDLE
	kernel32.SetProcessWorkingSetSizeEx.errcheck = errcheck_bool
	kernel32.SetProcessWorkingSetSizeEx.argtypes = (wintypes.HANDLE,
													ctypes.c_size_t,
													ctypes.c_size_t,
													wintypes.DWORD)
	hProcess = kernel32.GetCurrentProcess()
	kernel32.SetProcessWorkingSetSizeEx(hProcess, 204800, mem_limit_b, 6)
elif os.name=="posix":
	import resource
	resource.setrlimit(resource.RLIMIT_AS,(mem_limit_b,mem_limit_b))
else:
	raise RuntimeError("what system")

iisignature.rotinv2dprepare(14,"a")
print ("ok")
#input()