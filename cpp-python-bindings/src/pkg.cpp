#include <iostream>

// A simple class with a constuctor and some methods...

class Net
{
    public:
        Net(); // constructor _declaration_. Requires no type specification.
        void printVal();
        void setVal(int);
        int  getVal();
        void incVal();
        int val1;
    // private:
    //     int val1;
};

// class construction _definition_. Requires not ytpe specifiction.
Net::Net(){
    val1 = 1;
}

// define a method outside of the class definition
void Net::printVal()
{
    std::cout << "Value is " << val1 << std::endl;
}

// define a method outside of the class definition
void Net::setVal(int n)
{
    val1 = n;
}

int Net::getVal()
{
    return val1;
}

void Net::incVal()
{
    val1 += 1;
}


// Define C functions for the C++ class - as ctypes can only talk to C...
extern "C"
{   
    // we create a pointer to the object of type Net. The pointer type must be of the same type as the 
    // object/variable this pointer points to
    Net* CreateNet() {
        return new Net();
    } 

    // this func takes a pointer to the object of type Net and calls the bar() method of that object 
    // because this is a pointer, to access a member of the class, you use an arrow, not dot
    void method_1(Net* net) {
        net->printVal();
    }

    void method_2(Net* net, int n) {
        net->setVal(n);
    }

    int method_3(Net* net) {
        return net->getVal();
    }

    void method_4(Net* net) {
        net->incVal();
    }

}