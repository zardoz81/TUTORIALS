// #include <stdlib.h>
#include <math.h>
// #include <map>
#include <iostream>
#include <string>
#include <chrono>

using namespace std;

auto start = std::chrono::high_resolution_clock::now();
auto end_ = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> dur;

int iter = 0;
static const int N = 500;
const float tauexc = 10;
const float tauinh = 10;
const float winh = 1;
const float wmax = 27;
const float rho = 0.0025;
const float epsilon = 0.5;
const float dt = 0.1;
const float d = 5;
const float tauSTD =   500;
const float tauSTF =  200;
const float U =     0.6;
const float eta =     20;
const float tauw =    1000;
const float T =       4000;
float Sum = 0;
float W[N][N], delta[N][N];
float I[N], D[N], F[N], Iexc[N], Iinh[N], Iext[N], r[N];
float x = 0;

int main(){   
 


    // init W and D and F
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j) {
            W[i][j] = wmax * exp(-abs(double(i-j))/d);
        }
        W[i][i] = 0;
    }

    for (int i = 0; i < N; i++){
        D[i] = 1;
        F[i] = U;
    }

    for (float t = 0; t < T/dt; t+=dt) {
    
        // updIexc()
        for (int i = 0; i < N; i++) {
            Sum = 0;
            for (int j = 0; j < N; j++) {
                Sum += W[i][j] * r[j] * D[j] * F[j];
                Iexc[i] += (-Iexc[i]/tauexc + Sum) * dt;
            }
        }

        // updIinh()
        Sum = 0;
        for (int j = 0; j < N; j++) {
            Sum += r[j] * D[j] * F[j];
            Iinh[j] += (-Iinh[j]/tauinh + winh*Sum) * dt;
        }

        // updF()
        for (int j = 0; j < N; j++) {
            F[j] += ((U-F[j])/tauSTF + U*(1-F[j])*r[j]) * dt;
        }

        // updD()
        for (int j = 0; j < N; j++) {
            D[j] += ((1-D[j])/tauSTD - r[j]*D[j]*F[j]) * dt;
        }

        // updDelta()
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                delta[i][j] += (-delta[i][j] + eta*r[i]*r[j]*D[j]*F[j]) / tauw * dt;
            }
            delta[i][i] = 0;
        }

        // updW()
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                W[i][j] += delta[i][j];
            }
        }

        // updR()
        for (int j = 0; j < N; j++) {
            I[j] = Iexc[j] - Iinh[j] + Iext[j];
            x = rho * (I[j] - epsilon);
            if (x >= 0) {
                r[j] = x;
            } else {
                r[j] = 0;
            }
        }

        if (t < 10){
            for (int j = 0; j < 10; j++) {
                Iext[j] = 5;
            }
        }

        if (t > 10 && t < 11) {
            for (int j = 0; j < 10; j++) {
                Iext[j] = 0;
            }
        }

        if (t > 3000 && t <= 3010) {
            for (int j = 245; j < 255; j++) {
                Iext[j] = 5;
            }
        }

        if (t > 3010 && t < 3020) {
            for (int j = 245; j < 255; j++) {
                Iext[j] = 0;
            }
        }
        
        iter ++;

        if (iter%1000==0){
            end_ = std::chrono::high_resolution_clock::now();
            dur = end_ -start;
            cout << "Sim time: " << to_string(t) << "ms.\t Runtime: " << dur.count() << endl;
            start = std::chrono::high_resolution_clock::now();
        }
    }
    return 0;
}