#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <thread>
#include <tuple>
#include <algorithm>
#include <vector>
#include <chrono>


// 1D, 2D, 3D VECTORS
typedef std::vector<std::vector<std::vector<double>>> vvvd;
typedef std::vector<std::vector<double>> vvd;
typedef std::vector<double> vd;

// FLUID CONSTANTS
double gammac = 1.3;
double k = 150.0;
double R = 8.31;
double mu = 0.029;
double c = R / (mu * (gammac - 1));
double v_sound = 343.0;

// GRID STEPS
double dx = 0.001;
double dy = 0.001;
double dz = 0.001;


// CLASS FOR SYSTEM STATE
class SysState {
public:
    int n;
    vvvd RO;
    vvvd VX;
    vvvd VY;
    vvvd VZ;
    vvvd E;

    SysState(int n, double ro, double vx, double vy, double vz, double T) {
        this->n = n;
        this->RO = vvvd(n, vvd(n, vd(n, ro)));
        this->VX = vvvd(n, vvd(n, vd(n, vx)));
        this->VY = vvvd(n, vvd(n, vd(n, vy)));
        this->VZ = vvvd(n, vvd(n, vd(n, vz)));
        this->E = vvvd(n, vvd(n, vd(n, ro * ro * c * T)));
    }
};

// GET MAX ITEM OF 3D ARRAY
double maxi(const vvvd &arr) {
    double max_element = -10000000;
    for (int x = 0; x < arr.size(); x++)
        for (int y = 0; y < arr[x].size(); y++)
            for (int z = 0; z < arr[x][y].size(); z++)
                max_element = std::max(max_element, std::abs(arr[x][y][z]));
    return max_element;
}

// UPDATE ONE SLICE OF GRID
void update_slice(SysState &U, SysState &Unew, double dt, int i, int j) {

    int n = U.n;

    // DENSITY -- RO
    for (int x = 2; x <= n - 3; x += 2) {
        for (int y = 2; y <= n - 3; y += 2) {
            for (int z = i; z < j; z += 2) {
                Unew.RO[x][y][z] = U.RO[x][y][z]
                                    - dt * (U.RO[x][y][z] + U.RO[x + 2][y][z]) * U.VX[x + 1][y][z] / (2 * dx) +
                                    dt * (U.RO[x][y][z] + U.RO[x - 2][y][z]) * U.VX[x - 1][y][z] / (2 * dx)
                                    - dt * (U.RO[x][y][z] + U.RO[x][y + 2][z]) * U.VY[x][y + 1][z] / (2 * dy) +
                                    dt * (U.RO[x][y][z] + U.RO[x][y - 2][z]) * U.VY[x][y - 1][z] / (2 * dy)
                                    - dt * (U.RO[x][y][z] + U.RO[x][y][z + 2]) * U.VZ[x][y][z + 1] / (2 * dz) +
                                    dt * (U.RO[x][y][z] + U.RO[x][y][z - 2]) * U.VZ[x][y][z - 1] / (2 * dz)
                                    ;
            }
        }
    }

    // ENERGY -- FLOW
    for (int x = 2; x <= n - 3; x += 2) {
        for (int y = 2; y <= n - 3; y += 2) {
            for (int z = i; z < j; z += 2) {
                Unew.E[x][y][z] = U.E[x][y][z]
                                  -dt * gammac * (U.E[x][y][z] + U.E[x + 2][y][z]) * U.VX[x + 1][y][z] / (2 * dx) +
                                   dt * gammac * (U.E[x][y][z] + U.E[x - 2][y][z]) * U.VX[x - 1][y][z] / (2 * dx) -
                                   dt * gammac * (U.E[x][y][z] + U.E[x][y + 2][z]) * U.VY[x][y + 1][z] / (2 * dy) +
                                   dt * gammac * (U.E[x][y][z] + U.E[x][y - 2][z]) * U.VY[x][y - 1][z] / (2 * dy) -
                                   dt * gammac * (U.E[x][y][z] + U.E[x][y][z + 2]) * U.VZ[x][y][z + 1] / (2 * dz) +
                                   dt * gammac * (U.E[x][y][z] + U.E[x][y][z - 2]) * U.VZ[x][y][z - 1] / (2 * dz)
                                   ;
            }
        }
    }

    // ENERGY -- THERMAL
    for (int x = 2; x <= n - 3; x += 2) {
        for (int y = 2; y <= n - 3; y += 2) {
            for (int z = i; z < j; z += 2) {
                Unew.E[x][y][z] = U.E[x][y][z]
                        +dt * k * (U.E[x + 2][y][z] / (U.RO[x + 2][y][z] * c) - U.E[x][y][z] / (U.RO[x][y][z] * c)) / (dx*dx)
                        -dt * k * (U.E[x][y][z] / (U.RO[x][y][z] * c) - U.E[x - 2][y][z] / (U.RO[x - 2][y][z] * c)) / (dx*dx)
                        +dt * k * (U.E[x][y + 2][z] / (U.RO[x][y + 2][z] * c) - U.E[x][y][z] / (U.RO[x][y][z] * c)) / (dy*dy)
                        -dt * k * (U.E[x][y][z] / (U.RO[x][y][z] * c) - U.E[x][y - 2][z] / (U.RO[x][y - 2][z] * c)) / (dy*dy)
                        +dt * k * (U.E[x][y][z + 2] / (U.RO[x][y][z + 2] * c) - U.E[x][y][z] / (U.RO[x][y][z] * c)) / (dz*dz)
                        -dt * k * (U.E[x][y][z] / (U.RO[x][y][z] * c) - U.E[x][y][z - 2] / (U.RO[x][y][z - 2] * c)) / (dz*dz)
                        ;
            }
        }
    }

    // VELOCITY X
    for (int x = 3; x <= n - 4; x++) {
        for (int y = 2; y <= n - 3; y++) {
            for (int z = i; z < j; z++) {
                Unew.VX[x][y][z] = U.VX[x][y][z]
                        -dt * U.E[x + 1][y][z] * (gammac - 1) / (dx * (U.RO[x - 1][y][z] + U.RO[x + 1][y][z])) +
                        dt * U.E[x - 1][y][z] * (gammac - 1) / (dx * (U.RO[x - 1][y][z] + U.RO[x + 1][y][z]));
            }
        }
    }

    // VELOCITY Y
    for (int x = 2; x <= n - 3; x++) {
        for (int y = 3; y <= n - 4; y++) {
            for (int z = i; z < j; z++) {
                Unew.VY[x][y][z] = U.VY[x][y][z]
                        -dt * U.E[x][y + 1][z] * (gammac - 1) / (dy * (U.RO[x][y - 1][z] + U.RO[x][y + 1][z])) +
                        dt * U.E[x][y - 1][z] * (gammac - 1) / (dy * (U.RO[x][y - 1][z] + U.RO[x][y + 1][z]));
            }
        }
    }

    // VELOCITY Z
    for (int x = 2; x <= n - 3; x++) {
        for (int y = 2; y <= n - 3; y++) {
            for (int z = i + 1; z < std::min(j + 1, n - 2); z++) {
                Unew.VZ[x][y][z] += U.VZ[x][y][z]
                        -dt * U.E[x][y][z + 1] * (gammac - 1) / (dz * (U.RO[x][y][z - 1] + U.RO[x][y][z + 1])) +
                        dt * U.E[x][y][z - 1] * (gammac - 1) / (dz * (U.RO[x][y][z - 1] + U.RO[x][y][z + 1]));
            }
        }
    }
}

// PARALLEL UPDATE OF WHOLE GRID
void update_parallel(SysState &U, SysState &U_new, int n_threads){
    std::vector<std::thread> threads;
    int n = U.n;
    int d = ((n - 3) / 2) / n_threads;
    //double dt = 0.02 / (maxi(U.VX) / dx + maxi(U.VY) / dy + maxi(U.VZ) / dz + v_sound * std::sqrt(1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz)));
    double dt= 1e-8;
    int idx = 2;
    int r = 0;
    for (int i = 0; i < n_threads; i++){
        if (i < n_threads - 1) {
            r = idx + 2 * d;
        }
        else{
            r = n - 1;
        }
        threads.emplace_back(update_slice, std::ref(U), std::ref(U_new), dt, idx, r);
        idx = r;
    }
    for (auto &thr: threads){
        thr.join();
    }
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            U_new.VX[1][x][y] = 0;
            U_new.VX[n - 2][x][y] = 0;
            U_new.VY[x][1][y] = 0;
            U_new.VY[x][n - 2][y] = 0;
            U_new.VZ[x][y][n - 2] = 0;
            U_new.VZ[x][y][1] = 0;
        }
    }
}

// PERFORM MAC-CORMAC UPDATE FOR SLICE
void mac_cormac_slice(SysState &U, SysState &U_corrector, int i, int j){
    int n = U.n;
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            for (int z = i; z < j; z++) {
                U.RO[x][y][z] = 0.5 * (U.RO[x][y][z] + U_corrector.RO[x][y][z]);
                U.VX[x][y][z] = 0.5 * (U.VX[x][y][z] + U_corrector.VX[x][y][z]);
                U.VY[x][y][z] = 0.5 * (U.VY[x][y][z] + U_corrector.VY[x][y][z]);
                U.VZ[x][y][z] = 0.5 * (U.VZ[x][y][z] + U_corrector.VZ[x][y][z]);
                U.E[x][y][z] = 0.5 * (U.E[x][y][z] + U_corrector.E[x][y][z]);
            }
        }
    }
}

// PERFORM MAC-CORMAC UPDATE IN PARALLEL
void update_mac_cormac(SysState &U, SysState &U_predictor, SysState &U_corrector, int n_threads){
    int n = U.n;
    update_parallel(U, U_predictor, n_threads);
    update_parallel(U_predictor, U_corrector, n_threads);

    std::vector<std::thread> threads;
    
    int l = 0;
    int step = n / n_threads + 1;
    for (int i = 0; i < n_threads; i++){
        threads.emplace_back(mac_cormac_slice, std::ref(U), std::ref(U_corrector), l, std::min(n, l + step));
        l += step;
    }
    for (auto &thr: threads){
        thr.join();
    }
}


void read_init_state(std::string path, SysState &U, SysState &U_predictor, SysState &U_corrector){
    std::ifstream init_state(path);
    int x, y, z;
    double ro, vx, vy, vz, e;
    while (init_state >> x >> y >> z >> ro >> vx >> vy >> vz >> e){
        U.RO[x][y][z] = ro;
        U.VX[x][y][z] = vx;
        U.VY[x][y][z] = vy;
        U.VZ[x][y][z] = vz;
        U.E[x][y][z] = e;
        U_predictor.RO[x][y][z] = ro;
        U_predictor.VX[x][y][z] = vx;
        U_predictor.VY[x][y][z] = vy;
        U_predictor.VZ[x][y][z] = vz;
        U_predictor.E[x][y][z] = e;
        U_corrector.RO[x][y][z] = ro;
        U_corrector.VX[x][y][z] = vx;
        U_corrector.VY[x][y][z] = vy;
        U_corrector.VZ[x][y][z] = vz;
        U_corrector.E[x][y][z] = e;
    }
}


int main(int argc, char *argv[]) {

    ////////////////////////// PARSE CLI ARGUMENTS
    if (argc != 5) {
        std::cout << "DOUN" << std::endl;
        return 1;
    }
    int n = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int n_thr = std::stoi(argv[3]);
    std::string init_file = argv[4];

    /////////////////////////// INIT STATE
    SysState U{n, 0, 0, 0, 0, 0};
    SysState U_predictor{n, 0, 0, 0, 0, 0};
    SysState U_corrector{n, 0, 0, 0, 0, 0};
    read_init_state(init_file, U, U_predictor, U_corrector);

    ////////////////////////// TIME MEASUREMENT
    auto begin = std::chrono::system_clock::now();

    for (int i = 0; i < N; i++) {
        ////////////////////////// MAC-CORMAC
        update_mac_cormac(U, U_predictor, U_corrector, n_thr);
    }

    ////////////////////////// TIME MEASUREMENT
    auto end = std::chrono::system_clock::now();
    std::cout << double((end - begin).count()) / 1e9 << std::endl;

    return 0;
}


