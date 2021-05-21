#include <arrayfire.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <thread>
#include <tbb/concurrent_queue.h>
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
    SysState(const SysState &st){
        this->n = st.n;
        this->RO = st.RO.copy();
        this->VX = st.VX.copy();
        this->VY = st.VY.copy();
        this->VZ = st.VZ.copy();
        this->E = st.E.copy();
    }
    SysState(){
        int n = 1;
        this->n = n;
        this->RO = af::constant(0, n, n, n);
        this->VX = af::constant(0, n, n, n);
        this->VY = af::constant(0, n, n, n);
        this->VZ = af::constant(0, n, n, n);
        this->E = af::constant(0, n, n, n);
    }
    SysState operator+(const SysState &st){
        SysState result{n, 0, 0, 0, 0, 0};
        result.n = n;
        result.RO = RO + st.RO;
        result.VX = RO + st.VX;
        result.VY = RO + st.VY;
        result.VZ = RO + st.VZ;
        result.E = RO + st.E;

        return result;
    }
    SysState operator*(double k){
        SysState result{n, 0, 0, 0, 0, 0};
        result.n = n;
        result.RO = RO * k;
        result.VX = RO * k;
        result.VY = RO * k;
        result.VZ = RO * k;
        result.E = RO * k;

        return result;
    }
};


SysState update(const SysState &U){

    auto Unew = SysState(U);
    int n = Unew.n;

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

                    // + dt * U.RO(seq(2, n-3, 2)) * self.q[2:n-1:2, 2:n-1:2, 2:n-1:2]
                    // + dt * U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * (self.fx[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[3:n:2, 2:n-1:2, 2:n-1:2, 1] + U[1:n-2:2, 2:n-1:2, 2:n-1:2, 1]) / 2 
                    //                                         + self.fy[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[2:n-1:2, 3:n:2, 2:n-1:2, 2] + U[2:n-1:2, 1:n-2:2, 2:n-1:2, 2]) / 2 
                    //                                         + self.fz[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[2:n-1:2, 2:n-1:2, 3:n:2, 3] + U[2:n-1:2, 2:n-1:2, 1:n-2:2, 3]) / 2)
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

    return Unew;

}


void write_frame(SysState U, const std::string &path){
    std::ofstream file(path);
    int n = U.n;
    af::array vx = U.VX(seq(1, n - 2, 2), seq(2, n - 1, 2), seq(2, n - 1, 2));
    af::array vy = U.VY(seq(2, n - 1, 2), seq(1, n - 2, 2), seq(2, n - 1, 2));
    af::array vz = U.VZ(seq(2, n - 1, 2), seq(2, n - 1, 2), seq(1, n - 2, 2));
    af::array ep = U.E(seq(2, n-3, 2), seq(2, n-3, 2), seq(2, n-3, 2));

    int a = std::min(vx.dims()[0], ep.dims()[0]);
    int b = std::min(vx.dims()[1], ep.dims()[1]);
    int c = std::min(vx.dims()[2], ep.dims()[2]);
    int idx = 0;
    file << "x,y,z,vx,vy,vz,p\n";
    for (int ai = 0; ai < a; ai++){
        for (int bi = 0; bi < b; bi++){
            for (int ci = 0; ci < c; ci++){
                file << ai << "," << bi << "," << ci << "," << vx(ai, bi, ci).scalar<float>() << "," << vy(ai, bi, ci).scalar<float>() << "," << vz(ai, bi, ci).scalar<float>() << "," << ep(ai, bi, ci).scalar<float>() << "," << std::endl;
            }
        }
    }
    file.close();
}


void writer_fq(tbb::concurrent_queue<std::pair<SysState, std::string>> &queue){
    while (1){
        std::pair<SysState, std::string> p;
        auto ok = queue.try_pop(p);
        if (ok){
            auto U = p.first;
            auto path = p.second;

            if (path == "poison") return;
            write_frame(U, path);
        }
    }
}


int main(int argc, char *argv[]) {

    if (argc != 5){
        std::cout << "DOUN" << std::endl;
        return 1;
    }

    std::ios::sync_with_stdio(false);

    int n = std::stoi(argv[1]);
    SysState U{n, 1.25, 0, 0, 0, 300};
    U.E(n / 2 + 1, n / 2 + 1, n / 2 + 1) = 1.25 * 1.25 * c * 400;
    int N = std::stoi(argv[2]);
    int logfreq = std::stoi(argv[3]);
    int savefreq = std::stoi(argv[4]);

    tbb::concurrent_queue<std::pair<SysState, std::string>> write_queue;
    std::thread writer(writer_fq, std::ref(write_queue));
    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < N; i++){
        if ((i + 1) % logfreq == 0){
            std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
            std::cout << i + 1 << " " << logfreq / double(elapsed.count()) << std::endl;
            start = std::chrono::system_clock::now();
        }
        auto Upre = update(U);
        auto Ucor = update(Upre);
        U.RO = 0.5 * (Ucor.RO + U.RO);
        U.VX = 0.5 * (Ucor.VX + U.VX);
        U.VY = 0.5 * (Ucor.VY + U.VY);
        U.VZ = 0.5 * (Ucor.VZ + U.VZ);
        U.E = 0.5 * (Ucor.E + U.E);
        if ((i + 1) % savefreq == 0){
            write_queue.push(std::make_pair(SysState(U), "../data/exp_af_" + std::to_string(n) + "/_state_" + std::to_string(i / savefreq) + ".csv"));
        }
    }

    write_queue.push(std::make_pair(SysState(), "poison"));
    writer.join();

    return 0;
}