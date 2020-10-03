import ctypes

lib = ctypes.cdll.LoadLibrary('./pkg.so')

class cClassOne(object):

    # we have to specify the types of arguments and outputs of each function in the c++ class imported
    # the C types must match.

    def __init__(self):
        lib.CreateNet.argtypes = None # if the function gets no arguments
        lib.CreateNet.restype = ctypes.c_void_p

        lib.method_1.argtypes = [ctypes.c_void_p]
        lib.method_1.restype = ctypes.c_void_p


        lib.method_2.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.method_2.restype = ctypes.c_void_p

        lib.method_3.argtypes = [ctypes.c_void_p]
        lib.method_3.restype = ctypes.c_void_p

        lib.method_4.argtypes = [ctypes.c_void_p]
        lib.method_4.restype = ctypes.c_void_p

        # we call the constructor from the imported libpkg.so module
        self.obj = lib.CreateNet() # look in teh cpp code. CreateNet returns a pointer

    
    # in the Python wrapper, you can name these methods anything you want. Just make sure
    # you call the right C methods (that in turn call the right C++ methods)
    def printValToConsole(self):
        lib.method_1(self.obj)
    
    def setVal(self, val):
        lib.method_2(self.obj, val)

    def getVal(self):
        return lib.method_3(self.obj)

    def incVal(self):
        lib.method_4(self.obj)