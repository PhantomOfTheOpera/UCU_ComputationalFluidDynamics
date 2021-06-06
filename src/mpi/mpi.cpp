#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/optional/optional_io.hpp>
#include <thread>
#include <tbb/concurrent_queue.h>
#include <chrono>
#include <arrayfire.h>

namespace mpi = boost::mpi;

using af::span;
using af::seq;

class SysState {
private:
    static int globalID;
    constexpr static double gammac = 1.3, k = 150.0, R = 8.31, mu = 0.29, v_sound = 343.0,
            dx = 0.001, dy = 0.001, dz = 0.001;
    constexpr static double c = R / (mu * (gammac - 1));
    int n, n0, id, prevNode, nextNode, trueStart;
    const mpi::communicator compute;
    double *rightOut, *leftOut, *recvdData;
    af::array RO;
    af::array VX;
    af::array VY;
    af::array VZ;
    af::array E;

    int inline makeTag(int node, int i) {
        return 10*node + 1000*id + i;
    }

    void recvPartial(seq &&indexing, int node) {
        compute.irecv(node, makeTag(compute.rank(), 0), recvdData, 10*n*n).wait();
        RO(indexing, span, span) = af::array{2, n, n, recvdData}(span, span, span);
        VX(indexing, span, span) = af::array{2, n, n, recvdData + 2*n*n}(span, span, span);
        VY(indexing, span, span) = af::array{2, n, n, recvdData + 4*n*n}(span, span, span);
        VZ(indexing, span, span) = af::array{2, n, n, recvdData + 6*n*n}(span, span, span);
        E(indexing, span, span) = af::array{2, n, n, recvdData + 8*n*n}(span, span, span);
    }

    void sendPartial(seq &&indexing, int node, double* tempBuffer) {
        RO(indexing, span, span).host(tempBuffer);
        VX(indexing, span, span).host(tempBuffer + 2*n*n);
        VY(indexing, span, span).host(tempBuffer + 4*n*n);
        VZ(indexing, span, span).host(tempBuffer + 6*n*n);
        E(indexing, span, span).host(tempBuffer + 8*n*n);
        compute.isend(node, makeTag(node, 0), tempBuffer, 10*n*n);
    }

    void recvFirst() {
        if (compute.rank() == 0) {
            VX(1, span, span) = 0;
        } else {
            recvPartial(seq(0, 1), prevNode);
        }
        VY(span, 1, span) = 0;
        VZ(span, span, 1) = 0;
    }

    void recvLast() {
        if (compute.rank() == compute.size() - 1) {
            VX(n0-2, span, span) = 0;
        } else {
            recvPartial(seq(n0-2, n0-1), nextNode);
        }
        VY(span, n-2, span) = 0;
        VZ(span, span, n-2) = 0;
    }

    void sendReceive() {
        if (compute.rank() != 0) {
            sendPartial(seq(2, 3), prevNode, leftOut);
        }

        if (compute.rank() != compute.size() - 1) {
            sendPartial(seq(n0-4, n0-3), nextNode, rightOut);
        }


        recvFirst();
        recvLast();

        compute.barrier();
    }

public:
    SysState(int n_, int trueStart_, int n0_, double ro, double vx, double vy, double vz, double T,
             int prevNode_, int nextNode_, mpi::communicator& compute_) :
            n(n_), n0(n0_), id(globalID++), prevNode(prevNode_), nextNode(nextNode_), compute(compute_),
            recvdData(new double [10*n*n]), rightOut(new double [10*n*n]), leftOut(new double [10*n*n]),
            trueStart(trueStart_) {
        RO = af::constant(ro, n0, n, n, f64);
        VX = af::constant(vx, n0, n, n, f64);
        VY = af::constant(vy, n0, n, n, f64);
        VZ = af::constant(vz, n0, n, n, f64);
        E = af::constant(ro*ro*c*T, n0, n, n);
    }

    SysState(const SysState &st) :
            n(st.n), n0(st.n0), id(globalID++), prevNode(st.prevNode), nextNode(st.nextNode), compute(st.compute),
            recvdData(new double [10*n*n]), rightOut(new double [10*n*n]), leftOut(new double [10*n*n]),
            trueStart(st.trueStart) {
        RO = st.RO.copy();
        VX = st.VX.copy();
        VY = st.VY.copy();
        VZ = st.VZ.copy();
        E = st.E.copy();
        std::copy(st.recvdData, st.recvdData+10*n*n, recvdData);
        std::copy(st.leftOut, st.leftOut+10*n*n, leftOut);
        std::copy(st.rightOut, st.rightOut+10*n*n, rightOut);
    }

    SysState operator+(SysState st) const {
        st.RO += RO;
        st.VX += VX;
        st.VY += VY;
        st.VZ += VZ;
        st.E += E;

        return st;
    }

    SysState operator*(double k_) const {
        SysState result(*this);
        result.RO *= k_;
        result.VX *= k_;
        result.VY *= k_;
        result.VZ *= k_;
        result.E *= k_;

        return result;
    }

    void addNormalized(const SysState& frst, const SysState& scnd, double k_) {
        RO = k_ * (frst.RO + scnd.RO);
        VX = k_ * (frst.VX + scnd.VX);
        VY = k_ * (frst.VY + scnd.VY);
        VZ = k_ * (frst.VZ + scnd.VZ);
        E = k_ * (frst.E + scnd.E);
    }

    void update(const SysState &U);

    void setE(size_t x, size_t y, size_t z, double ro_, double T) {
        if (x >= trueStart && x <= trueStart + n0) {
            E(x-trueStart, y, z) = ro_ * ro_ * c * T;
        }
    }

    virtual ~SysState();
};


int SysState::globalID = 0;


void SysState::update(const SysState &U) {
    sendReceive();

    std::array<double, 3> tMax{af::max<double>(af::abs(U.VX)), af::max<double>(af::abs(U.VY)), af::max<double>(af::abs(U.VZ))};
    for (double & mx : tMax) {
        mpi::all_reduce(compute, mx, mpi::maximum<double>());
    }
    double dt = 0.005 / (tMax[0] / dx + tMax[1] / dy + tMax[2] / dz + v_sound * std::sqrt(1 / (dx*dx) + 1 / (dy*dy) + 1 / (dz*dz)));

    RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) = (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2))
                                                           // X DENSITY CHANGE
                                                           - dt * (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(4, n0-1, 2), seq(2, n-3, 2), seq(2, n-3, 2))) * U.VX(seq(3, n0-2, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (dx * 2)
                                                           + dt * (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(0, n0-5, 2), seq(2, n-3, 2), seq(2, n-3, 2))) * U.VX(seq(1, n0-4, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (dx * 2)
                                                           // Y DENSITY CHANGE
                                                           - dt * (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(2, n0-3, 2), seq(4, n-1, 2), seq(2, n-3, 2))) * U.VY(seq(2, n0-3, 2), seq(3, n-2, 2), seq(2, n-3, 2)) / (dy * 2)
                                                           + dt * (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(2, n0-3, 2), seq(0, n-5, 2), seq(2, n-3, 2))) * U.VY(seq(2, n0-3, 2), seq(1, n-4, 2), seq(2, n-3, 2)) / (dy * 2)
                                                           // Z DENSITY CHANGE
                                                           - dt * (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(4, n-1, 2))) * U.VZ(seq(2, n0-3, 2), seq(2, n-3, 2), seq(3, n-2, 2)) / (dz * 2)
                                                           + dt * (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(0, n-5, 2))) * U.VZ(seq(2, n0-3, 2), seq(2, n-3, 2), seq(1, n-4, 2)) / (dz * 2)
    );

    E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) = (U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2))
                                                          // X ENERGY CHANGE
                                                          - dt * gammac * (U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(4, n0-1, 2), seq(2, n-3, 2), seq(2, n-3, 2))) * U.VX(seq(3, n0-2, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (dx * 2)
                                                          + dt * gammac * (U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(0, n0-5, 2), seq(2, n-3, 2), seq(2, n-3, 2))) * U.VX(seq(1, n0-4, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (dx * 2)
                                                          // Y ENERGY CHANGE
                                                          - dt * gammac * (U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(2, n0-3, 2), seq(4, n-1, 2), seq(2, n-3, 2))) * U.VY(seq(2, n0-3, 2), seq(3, n-2, 2), seq(2, n-3, 2)) / (dy * 2)
                                                          + dt * gammac * (U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(2, n0-3, 2), seq(0, n-5, 2), seq(2, n-3, 2))) * U.VY(seq(2, n0-3, 2), seq(1, n-4, 2), seq(2, n-3, 2)) / (dy * 2)
                                                          // Z ENERGY CHANGE
                                                          - dt * gammac * (U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(4, n-1, 2))) * U.VZ(seq(2, n0-3, 2), seq(2, n-3, 2), seq(3, n-2, 2)) / (dz * 2)
                                                          + dt * gammac * (U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(0, n-5, 2))) * U.VZ(seq(2, n0-3, 2), seq(2, n-3, 2), seq(1, n-4, 2)) / (dz * 2)

            // + dt * U.RO(seq(2, n-3, 2)) * self.q[2:n-1:2, 2:n-1:2, 2:n-1:2]
            // + dt * U[2:n-1:2, 2:n-1:2, 2:n-1:2, 0] * (self.fx[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[3:n:2, 2:n-1:2, 2:n-1:2, 1] + U[1:n-2:2, 2:n-1:2, 2:n-1:2, 1]) / 2
            //                                         + self.fy[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[2:n-1:2, 3:n:2, 2:n-1:2, 2] + U[2:n-1:2, 1:n-2:2, 2:n-1:2, 2]) / 2
            //                                         + self.fz[2:n-1:2, 2:n-1:2, 2:n-1:2] * (U[2:n-1:2, 2:n-1:2, 3:n:2, 3] + U[2:n-1:2, 2:n-1:2, 1:n-2:2, 3]) / 2)
    );

    E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) = (E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2))
                                                          // X THERMAL CHANGE
                                                          + dt * k * (U.E(seq(4, n0-1, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(4, n0-1, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c) - U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dx*dx)
                                                          + dt * k * (U.E(seq(0, n0-5, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(0, n0-5, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c) - U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dx*dx)
                                                          // Y THERMAL CHANGE
                                                          + dt * k * (U.E(seq(2, n0-3, 2), seq(4, n-1, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n0-3, 2), seq(4, n-1, 2), seq(2, n-3, 2)) * c) - U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dy*dy)
                                                          + dt * k * (U.E(seq(2, n0-3, 2), seq(0, n-5, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n0-3, 2), seq(0, n-5, 2), seq(2, n-3, 2)) * c) - U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dy*dy)
                                                          // Z THERMAL CHANGE
                                                          + dt * k * (U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(4, n-1, 2)) / (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(4, n-1, 2)) * c) - U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dz*dz)
                                                          + dt * k * (U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(0, n-5, 2)) / (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(0, n-5, 2)) * c) - U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) / (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * c)) / (dz*dz)
    );

    // X VELOCITY CHANGE
    auto p1_x = U.E(seq(4, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * (gammac - 1);
    auto p2_x = U.E(seq(2, n0-5, 2), seq(2, n-3, 2), seq(2, n-3, 2)) * (gammac - 1);
    VX(seq(3, n0-4, 2), seq(2, n-3, 2), seq(2, n-3, 2)) = (U.VX(seq(3, n0-4, 2), seq(2, n-3, 2), seq(2, n-3, 2))
                                                           - dt * p1_x / (dx * (U.RO(seq(2, n0-5, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(4, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2))))
                                                           + dt * p2_x / (dx * (U.RO(seq(2, n0-5, 2), seq(2, n-3, 2), seq(2, n-3, 2)) + U.RO(seq(4, n0-3, 2), seq(2, n-3, 2), seq(2, n-3, 2))))
    );

    // Y VELOCITY CHANGE
    auto p1_y = U.E(seq(2, n0-3, 2), seq(4, n-3, 2), seq(2, n-3, 2)) * (gammac - 1);
    auto p2_y = U.E(seq(2, n0-3, 2), seq(2, n-5, 2), seq(2, n-3, 2)) * (gammac - 1);
    VY(seq(2, n0-3, 2), seq(3, n-4, 2), seq(2, n-3, 2)) = (U.VY(seq(2, n0-3, 2), seq(3, n-4, 2), seq(2, n-3, 2))
                                                           - dt * p1_y / (dy * (U.RO(seq(2, n0-3, 2), seq(2, n-5, 2), seq(2, n-3, 2)) + U.RO(seq(2, n0-3, 2), seq(4, n-3, 2), seq(2, n-3, 2))))
                                                           + dt * p2_y / (dy * (U.RO(seq(2, n0-3, 2), seq(2, n-5, 2), seq(2, n-3, 2)) + U.RO(seq(2, n0-3, 2), seq(4, n-3, 2), seq(2, n-3, 2))))
    );
    // Z VELOCITY CHANGE
    auto p1_z = U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(4, n-3, 2)) * (gammac - 1);
    auto p2_z = U.E(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-5, 2)) * (gammac - 1);
    VZ(seq(2, n0-3, 2), seq(2, n-3, 2), seq(3, n-4, 2)) = (U.VZ(seq(2, n0-3, 2), seq(2, n-3, 2), seq(3, n-4, 2))
                                                           - dt * p1_z / (dz * (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-5, 2)) + U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(4, n-3, 2))))
                                                           + dt * p2_z / (dz * (U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(2, n-5, 2)) + U.RO(seq(2, n0-3, 2), seq(2, n-3, 2), seq(4, n-3, 2))))
    );
}


SysState::~SysState() {
    delete[] recvdData;
    delete[] rightOut;
    delete[] leftOut;
}


int main(int argc, char *argv []) {
    mpi::environment env{argc, argv, mpi::threading::funneled};
    mpi::communicator compute;

    if (argc != 4) {
        std::cout << "DOUN" << std::endl;
        env.abort(1);
    }

    std::ios::sync_with_stdio(false);

    int n = std::stoi(argv[1]);
    int timeN = std::stoi(argv[2]);
    std::string backend = argv[3];

    /////////////////////////// SELECT BACKEND
    if (backend == "CPU"){
        af::setBackend(AF_BACKEND_CPU);
    }
    else if (backend == "OPENCL_GPU"){
        af::setBackend(AF_BACKEND_OPENCL);
        af::setDevice(compute.rank());
    }
    else if (backend == "CUDA"){
        af::setBackend(AF_BACKEND_CUDA);
        af::setDevice(compute.rank());
    }
    else{
        std::cout << "Backend not supported" << std::endl;
    }

    int strip_len;
    int start_i = 0;
    for (size_t i = 0; i < compute.rank(); i++) {
        start_i += static_cast<int>(i == 0 ? n - std::round(n / compute.size()) * (compute.size() - 1)
                : std::round(n / compute.size())) - 2;
    }
    strip_len = static_cast<int>(compute.rank() == 0 ? n - std::round(n / compute.size()) * (compute.size() - 1)
            : std::round(n / compute.size()));
    strip_len = compute.rank() == 0 || compute.rank() == compute.size()-1 ? strip_len + 2 : strip_len + 4;
    int next_node = (compute.rank()+1) % compute.size();
    int prev_node = (compute.rank()+compute.size()-1) % compute.size();

    SysState U{n, start_i, strip_len, 1.25, 0, 0, 0, 300, prev_node, next_node, compute}, Upre(U), Ucor(U);
    U.setE(n / 2 , n / 2, n / 2, 1.25, 400);

    compute.barrier();
    auto begin = std::chrono::system_clock::now();
    for (int timeI = 0; timeI < timeN; timeI++) {
        Upre.update(U);
        Ucor.update(Upre);
        U.addNormalized(Upre, Ucor, 0.5);
    }
    compute.barrier();
    auto end = std::chrono::system_clock::now();
    if (compute.rank() == 0){
        std::cout << double((end - begin).count()) / 1e9 << std::endl;
    }
}