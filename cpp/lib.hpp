#ifndef LIB

#define LIB

#include <vector>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <random>
#include <iomanip>
#include <map>

using namespace std;

// type definitions:
// typedef vector<vector<double> > Array2D;
// typedef vector<double> Array1D;

// Array1D Make1Darray(int sz) {
// 	Array1D arr (sz, 0);
// 	return arr;
// }

// Array2D Make2DArray(int height, int width){
//     Array2D arr (height, Array1D(width, 0));
//     return arr;
// }

// Array2D rnd(int sz0, int sz1, double min , double max){
//     random_device rd;
//     mt19937 gen(rd());
//     uniform_real_distribution <> dis(min, max);
//     Array2D arr = Make2DArray(sz0, sz1);
//     for (int i = 0; i < sz0; ++i) {
//         for (int j = 0; j < sz1; ++j) {
//             arr[i][j] = dis(gen);
//         }
//     }
//     return arr;
// }


class Net
{   
    public:
        // int N;
        // Array2D W, delta;
        // Array1D I, D, F, Iexc, Iinh, Iext, r;
        static const int N = 500;
        float W[N][N];
        float delta[N][N];
        float d, rho, epsilon, wmax, tauexc, tauinh, dt, winh, tauSTD, tauSTF, U, eta, tauw, T, Sum;
        float I[N], D[N], F[N], Iexc[N], Iinh[N], Iext[N], r[N];
        
        //class constructor
        Net() {
            // N = 500;
            tauexc = 10;
            tauinh = 10;
            winh = 1;
            wmax = 27;
            rho = 0.0025;
            epsilon = 0.5;
            dt = 0.1;
            d = 5;
            tauSTD =   500;
            tauSTF =  200;
            U =     0.6;
            eta =     20;
            tauw =    1000;
            T =       4000;
            Sum = 0;
            
            initW();
            // delta = Make2DArray(N, N);
            // r = Make1Darray(N);
            // D = Make1Darray(N);
            // F = Make1Darray(N);
            // delta = Make2Darray(N, N);
            // I = Make1Darray(N);
            // Iexc = Make1Darray(N);
            // Iinh = Make1Darray(N);
            // Iext = Make1Darray(N);
            for (int i = 0; i < N; i++){
                D[i] = 1;
                F[i] = U;
            }
        }

        inline void initW() {
            // W = Make2DArray(N, N);
            for (int i = 0; i < N; ++i){
                for (int j = 0; j < N; ++j) {
                    W[i][j] = wmax * exp(-abs(double(i-j))/d);
                }
                W[i][i] = 0;
            }
        }

        inline void frate() {
            for (int j = 0; j < N; j++) {
                I[j] = Iexc[j] - Iinh[j] + Iext[j];
                float x = rho * (I[j] - epsilon);
                if (x >= 0) {
                    r[j] = x;
                } else {
                    r[j] = 0;
                }
            }
        }

        inline void updIexc() {
            for (int i = 0; i < N; i++) {
                Sum = 0;
                for (int j = 0; j < N; j++) {
                    Sum += W[i][j] * r[j] * D[j] * F[j];
                    Iexc[i] += (-Iexc[i]/tauexc + Sum) * dt;
                }
            }
        }

        inline void updIinh() {
            Sum = 0;
            for (int j = 0; j < N; j++) {
                Sum += r[j] * D[j] * F[j];
                Iinh[j] += (-Iinh[j]/tauinh + winh*Sum) * dt;
            }
        }

        inline void updD() {
            for (int j = 0; j < N; j++) {
                D[j] += ((1-D[j])/tauSTD - r[j]*D[j]*F[j]) * dt;
            }
        }

        inline void updF() {
            for (int j = 0; j < N; j++) {
                F[j] += ((U-F[j])/tauSTF + U*(1-F[j])*r[j]) * dt;
            }
        }

        inline void updDelta() {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    delta[i][j] += (-delta[i][j] + eta*r[i]*r[j]*D[j]*F[j]) / tauw * dt;
                }
                delta[i][i] = 0;
            }
        }

        inline void updW() {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    W[i][j] += delta[i][j];
                }
            }
        }

        inline void updR() {
                frate();
            }
        
        inline void step(int n) {
            for (int i=0; i<n; i++){
                updIexc();
                updIinh();
                updF();
                updD();
                updDelta();
                updW();
                updR();
            }
        }



};
#endif
