#include "lib.hpp"
#include <iostream>
#include <string>
#include <chrono>


// using namespace std;

/*
https://code.visualstudio.com/docs/cpp/config-clang-mac
*/

int main()
{
    Net fred;

    int iter = 0;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur;
    
    for (float t = 0; t < fred.T/fred.dt; t++) {

        if (t <= 10) {
            for (int j = 0; j < 10; j++) {
                fred.Iext[j] = 5;
            }
        }

        if (t > 10 && t < 11) {
            for (int j = 0; j < 10; j++) {
                fred.Iext[j] = 0;
            }
        }

        if (t > 3000 && t <= 3010) {
            for (int j = 245; j < 255; j++) {
                fred.Iext[j] = 5;
            }
        }

        if (t > 3010 && t < 3020) {
            for (int j = 245; j < 255; j++) {
                fred.Iext[j] = 0;
            }
        }
        
        fred.step(1);
        iter ++;

        if (iter%1000==0){
            end = std::chrono::high_resolution_clock::now();
            dur = end-start;
            cout << "Sim time: " << to_string(int(t)) << "ms.\t Runtime: " << dur.count() << endl;
            start = std::chrono::high_resolution_clock::now();
        }

    }

    return 0;
}
