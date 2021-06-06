#include <arrayfire.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <chrono>

using af::span;
using af::seq;

double gammac = 1.3;
double k = 150.0;
double R = 8.31;
double mu = 0.29;
double c = R / (mu * (gammac - 1));
double v_sound = 343.0;

double dx = 0.001;
double dy = 0.001;
double dz = 0.001;

class SysState {
public:
    int n;
    af::array RO;
    af::array VX;
    af::array VY;
    af::array VZ;
    af::array E;
    SysState(int n, double ro, double vx, double vy, double vz, double T){
        this->n = n;
        this->RO = af::constant(ro, n, n, n);
        this->VX = af::constant(vx, n, n, n);
        this->VY = af::constant(vy, n, n, n);
        this->VZ = af::constant(vz, n, n, n);
        this->E = af::constant(ro*ro*c*T, n, n, n);
    }

    void operator+=(const SysState &st){
        RO = RO + st.RO;
        VX = VX + st.VX;
        VY = VY + st.VY;
        VZ = VZ + st.VZ;
        E = E + st.E;
    }
    void operator*=(double k){
        RO = RO * k;
        VX = RO * k;
        VY = RO * k;
        VZ = RO * k;
        E = RO * k;
    }
};


void update(SysState &U, SysState &Unew){

    int n = U.n;
    double dt = 0.005 / (af::max<double>(af::abs(U.VX)) / dx + af::max<double>(af::abs(U.VY)) / dy + af::max<double>(af::abs(U.VZ)) / dz + v_sound * std::sqrt(1 / (dx*dx) + 1 / (dy*dy) + 1 / (dz * dz)));

    Unew.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) = (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2))
                    // X DENSITY CHANGE
                    - dt * (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(4, n-1, 2), seq(2, n-3, 2), seq(2, n-3, 2))) * U.VX(seq(3, n-2, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (dx * 2)
                    + dt * (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(0, n-5, 2), seq(2, n-3, 2), seq(2, n-3, 2))) * U.VX(seq(1, n-4, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (dx * 2)
                    // Y DENSITY CHANGE
                    - dt * (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(2, n-3, 2), seq(4, n-1, 2), seq(2, n-3, 2))) * U.VY(seq(2, n-3, 2), seq(3, n-2, 2), seq(2, n-3, 2)) / (dy * 2)
                    + dt * (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(2, n-3, 2), seq(0, n-5, 2), seq(2, n-3, 2))) * U.VY(seq(2, n-3, 2), seq(1, n-4, 2), seq(2, n-3, 2)) / (dy * 2)
                    // Z DENSITY CHANGE
                    - dt * (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(4, n-1, 2))) * U.VZ(seq(2, n-3, 2), seq(2, n-3, 2), seq(3, n-2, 2)) / (dz * 2)
                    + dt * (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(0, n-5, 2))) * U.VZ(seq(2, n-3, 2), seq(2, n-3, 2), seq(1, n-4, 2)) / (dz * 2)
            );

    Unew.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) = (U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2))
                    // X ENERGY CHANGE
                    - dt * gammac * (U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(4, n-1, 2), seq(2, n-3, 2), seq(2, n-3, 2))) * U.VX(seq(3, n-2, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (dx * 2)
                    + dt * gammac * (U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(0, n-5, 2), seq(2, n-3, 2), seq(2, n-3, 2))) * U.VX(seq(1, n-4, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (dx * 2)
                    // Y ENERGY CHANGE
                    - dt * gammac * (U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(2, n-3, 2), seq(4, n-1, 2), seq(2, n-3, 2))) * U.VY(seq(2, n-3, 2), seq(3, n-2, 2), seq(2, n-3, 2)) / (dy * 2)
                    + dt * gammac * (U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(2, n-3, 2), seq(0, n-5, 2), seq(2, n-3, 2))) * U.VY(seq(2, n-3, 2), seq(1, n-4, 2), seq(2, n-3, 2)) / (dy * 2)
                    // Z ENERGY CHANGE
                    - dt * gammac * (U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(4, n-1, 2))) * U.VZ(seq(2, n-3, 2), seq(2, n-3, 2), seq(3, n-2, 2)) / (dz * 2)
                    + dt * gammac * (U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(0, n-5, 2))) * U.VZ(seq(2, n-3, 2), seq(2, n-3, 2), seq(1, n-4, 2)) / (dz * 2)
            );
    
    Unew.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) = (Unew.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2))
                    // X THERMAL CHANGE
                    + dt * k * (U.E(seq(4, n-1, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(4, n-1, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c) - U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dx*dx)
                    + dt * k * (U.E(seq(0, n-5, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(0, n-5, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c) - U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dx*dx)
                    // Y THERMAL CHANGE
                    + dt * k * (U.E(seq(2, n-3, 2), seq(4, n-1, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n-3, 2), seq(4, n-1, 2), seq(2, n-3, 2)) * c) - U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dy*dy)
                    + dt * k * (U.E(seq(2, n-3, 2), seq(0, n-5, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n-3, 2), seq(0, n-5, 2), seq(2, n-3, 2)) * c) - U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dy*dy)
                    // Z THERMAL CHANGE
                    + dt * k * (U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(4, n-1, 2)) / (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(4, n-1, 2)) * c) - U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dz*dz)
                    + dt * k * (U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(0, n-5, 2)) / (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(0, n-5, 2)) * c) - U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dz*dz)
            );
    
    // X VELOCITY CHANGE
    auto p1_x = U.E(seq(4, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * (gammac - 1);
    auto p2_x = U.E(seq(2, n-5, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * (gammac - 1);
    Unew.VX(seq(3, n-4, 2), seq(2, n-3, 2), seq(2, n-3, 2)) = (U.VX(seq(3, n-4, 2), seq(2, n-3, 2), seq(2, n-3, 2))
               - dt * p1_x / (dx * (U.RO(seq(2, n-5, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(4, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2))))
               + dt * p2_x / (dx * (U.RO(seq(2, n-5, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(4, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2))))
    );

    // Y VELOCITY CHANGE
    auto p1_y = U.E(seq(2, n-3, 2), seq(4, n-3, 2), seq(2, n-3, 2)) * (gammac - 1);
    auto p2_y = U.E(seq(2, n-3, 2), seq(2, n-5, 2), seq(2, n-3, 2)) * (gammac - 1);
    Unew.VY(seq(2, n-3, 2), seq(3, n-4, 2), seq(2, n-3, 2)) = (U.VY(seq(2, n-3, 2), seq(3, n-4, 2), seq(2, n-3, 2))
               - dt * p1_y / (dy * (U.RO(seq(2, n-3, 2), seq(2, n-5, 2), seq(2, n-3, 2)) + U.RO(seq(2, n-3, 2), seq(4, n-3, 2), seq(2, n-3, 2))))
               + dt * p2_y / (dy * (U.RO(seq(2, n-3, 2), seq(2, n-5, 2), seq(2, n-3, 2)) + U.RO(seq(2, n-3, 2), seq(4, n-3, 2), seq(2, n-3, 2))))
    );
    // Z VELOCITY CHANGE
    auto p1_z = U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(4, n-3, 2)) * (gammac - 1);
    auto p2_z = U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-5, 2)) * (gammac - 1);
    Unew.VZ(seq(2, n-3, 2), seq(2, n-3, 2), seq(3, n-4, 2)) = (U.VZ(seq(2, n-3, 2), seq(2, n-3, 2), seq(3, n-4, 2))
               - dt * p1_z / (dz * (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-5, 2)) + U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(4, n-3, 2))))
               + dt * p2_z / (dz * (U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-5, 2)) + U.RO(seq(2, n-3, 2), seq(2, n-3, 2), seq(4, n-3, 2))))
    );

    Unew.VX(1, span, span) = 0;
    Unew.VX(n-2, span, span) = 0;

    Unew.VY(span, 1, span) = 0;
    Unew.VY(span, n-2, span) = 0;

    Unew.VZ(span, span, 1) = 0;
    Unew.VZ(span, span, n-2) = 0;
}


void read_init_state(const char* path, SysState &U, SysState &U_predictor, SysState &U_corrector){

    U.RO = af::readArray(path, "ro");
    U.RO = af::readArray(path, "vx");
    U.RO = af::readArray(path, "vy");
    U.RO = af::readArray(path, "vz");
    U.RO = af::readArray(path, "e");
    U_predictor.RO = af::readArray(path, "ro");
    U_predictor.RO = af::readArray(path, "vx");
    U_predictor.RO = af::readArray(path, "vy");
    U_predictor.RO = af::readArray(path, "vz");
    U_predictor.RO = af::readArray(path, "e");
    U_corrector.RO = af::readArray(path, "ro");
    U_corrector.RO = af::readArray(path, "vx");
    U_corrector.RO = af::readArray(path, "vy");
    U_corrector.RO = af::readArray(path, "vz");
    U_corrector.RO = af::readArray(path, "e");
    
}


int main(int argc, char *argv[]) {

    ////////////////////////// PARSE CLI ARGUMENTS
    if (argc != 5) {
        std::cout << "DOUN" << std::endl;
        return 1;
    }
    int n = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    std::string init_file = argv[3];
    std::string backend = argv[4];

    /////////////////////////// SELECT BACKEND
    if (backend == "CPU"){
        af::setBackend(AF_BACKEND_CPU);
    }
    else if (backend == "OPENCL_CPU"){
        af::setBackend(AF_BACKEND_OPENCL);
        af::setDevice(2);
    }
    else if (backend == "OPENCL_GPU"){
        af::setBackend(AF_BACKEND_OPENCL);
        af::setDevice(0);
    }
    else if (backend == "CUDA"){
        af::setBackend(AF_BACKEND_CUDA);
    }
    else{
        std::cout << "Backend not supported" << std::endl;
    }

    /////////////////////////// INIT STATE
    SysState U{n, 0, 0, 0, 0, 0};
    SysState U_predictor{n, 0, 0, 0, 0, 0};
    SysState U_corrector{n, 0, 0, 0, 0, 0};
    read_init_state(init_file.c_str(), U, U_predictor, U_corrector);

    ////////////////////////// TIME MEASUREMENT
    auto begin = std::chrono::system_clock::now();

    for (int i = 0; i < N; i++){
        update(U, U_predictor);
        update(U_predictor, U_corrector);
        U += U_corrector;
        U *= 0.5;
    }

    ////////////////////////// TIME MEASUREMENT
    auto end = std::chrono::system_clock::now();
    std::cout << double((end - begin).count()) / 1e9 << std::endl;

    return 0;
}