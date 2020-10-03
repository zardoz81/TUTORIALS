# we import a Python wrapper class (that interfaces with the C++ code 
# through the C wrapper methods)
from cClasses import cClassOne

# We'll create a Foo object with a value of 5...
obj = cClassOne()

print(obj.getVal())

obj.setVal(1)

for i in range(100):
    obj.incVal()

print(obj.getVal())

