import os, sys
#this so that we can import a test file from this directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import unittest
import test_sig

def my_module_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_sig)
    return suite
