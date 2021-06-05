#include <CL/sycl.hpp>
#include <array>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>


// output message for runtime exceptions
#define EXCEPTION_MSG \
  "    If you are targeting an FPGA hardware, please ensure that an FPGA board is plugged to the system, \n\
        set up correctly and compile with -DFPGA  \n\
    If you are targeting the FPGA emulator, compile with -DFPGA_EMULATOR.\n"

namespace scl = cl::sycl;

struct MainParams
{
    const double gamma = 7.0 / 5.0;
    const double k = 300;
    const double R = 8.31;
    const double mu = 0.029;
    const double c = R / ((gamma - 1) * mu);
    const double v_sound = 343;
    double dx = 0.001;
    double dy = 0.001;
    double dz = 0.001;
};

using Buffer3D = scl::buffer<double, 3>;

// Convience data access definitions
constexpr scl::access::mode dp_read = scl::access::mode::read;
constexpr scl::access::mode dp_write = scl::access::mode::discard_write;
constexpr scl::access::mode read_write = scl::access::mode::read_write;

// key parmeters
constexpr size_t params = 5;
constexpr size_t n = 203;

template <typename T> using Tensor3D = std::array<T, n* n* n>;
template <typename T, size_t N> using Tensor4D = std::array<Tensor3D<T>, N>;

template <typename T>
static void fillTensor3D(Tensor3D<T>& tensor, T value) {
    for (size_t i = 0; i < n; i++) {
        for (size_t ii = 0; ii < n; ii++) {
            for (size_t iii = 0; iii < n; iii++) {
                tensor[i * n * n + ii * n + iii] = value;
            }
        }
    }
}

template <typename T, size_t N>
static void fillTensor4D(Tensor4D<T, N>& tensor, std::array<T, N>&& values) {
    for (size_t iv = 0; iv < N; iv++) {
        fillTensor3D<T>(tensor[iv], values[iv]);
    }
}

template <size_t N>
void add4DTensorsScaled(scl::queue& deviceQ, std::vector<Buffer3D>& result, std::vector<Buffer3D>& lhs, std::vector<Buffer3D>& rhs, scl::range<3>& num_items, double scale = 0.5) {
    for (size_t param = 0; param < N; param++) {
        deviceQ.submit([&](scl::handler& h) {
            auto lhs_acc = lhs[param].get_access<dp_read, scl::access::target::global_buffer>(h);
            auto rhs_acc = rhs[param].get_access<dp_read, scl::access::target::global_buffer>(h);

            auto sum_acc = result[param].get_access<dp_write, scl::access::target::global_buffer>(h);

            h.parallel_for<class vector_addition>(num_items, [=](scl::id<3> i) {
                sum_acc[i] = scale * (lhs_acc[i] + rhs_acc[i]);
                });
            });
    }
}

//double getAbsMax(scl::queue& deviceQ, std::vector<Buffer3D>& tensor, scl::range<3> num_items, size_t last_id) {
//    double max_val = 0.0;
//
//}

double getAbsMax(scl::queue& deviceQ, std::vector<Buffer3D>& tensor, scl::range<3> num_items, size_t last_id) {
    double max_val = 0.0;
    auto tensor_acc = tensor[last_id].get_access<dp_read>();
    for (size_t i = 0; i < n; i += 4) {
        for (size_t ii = 0; ii < n; ii += 4) {
            for (size_t iii = 0; iii < n; iii += 4) {
                if (abs(tensor_acc[i][ii][iii]) > max_val) {
                    max_val = abs(tensor_acc[i][ii][iii]);
                }
            }
        }
    }
    return max_val;
}

template <size_t N>
void update_RO(scl::queue& deviceQ, std::vector<Buffer3D>& Unew, std::vector<Buffer3D>& Uold, double dt, MainParams mp) {
    scl::range<3> num_items{ n, n, n };

    deviceQ.submit([&](scl::handler& h) {
        auto Uold_acc = Uold[0].get_access<dp_read>(h);
        auto Unew_acc = Unew[0].get_access<dp_write>(h);
        auto x_acc = Uold[1].get_access<dp_read>(h);
        auto y_acc = Uold[2].get_access<dp_read>(h);
        auto z_acc = Uold[3].get_access<dp_read>(h);
        auto dx = mp.dx, dy = mp.dy, dz = mp.dz;

        h.parallel_for<class ro_update>(num_items, [=](scl::id<3> i) {
            Unew_acc[i] = Uold_acc[i];
            // TODO likely 1/2 performance, need to fix!
            if (i[0] % 2 == 1 && i[0] >= 2 && i[0] <= (n - 1)
                && i[1] % 2 == 1 && i[1] >= 2 && i[1] <= (n - 1)
                && i[2] % 2 == 1 && i[2] >= 2 && i[2] <= (n - 1)) {
                Unew_acc[i] = (Uold_acc[i]
                    - dt * (Uold_acc[i] + Uold_acc[i[0] + 2][i[1]][i[2]] * x_acc[i[0] + 1][i[1]][i[2]]) / (dx * 2)
                    + dt * (Uold_acc[i] + Uold_acc[i[0] - 2][i[1]][i[2]] * x_acc[i[0] - 1][i[1]][i[2]]) / (dx * 2)
                    - dt * (Uold_acc[i] + Uold_acc[i[0]][i[1] + 2][i[2]] * y_acc[i[0]][i[1] + 1][i[2]]) / (dy * 2)
                    + dt * (Uold_acc[i] + Uold_acc[i[0]][i[1] - 2][i[2]] * y_acc[i[0]][i[1] - 1][i[2]]) / (dy * 2)
                    - dt * (Uold_acc[i] + Uold_acc[i[0]][i[1]][i[2] + 2] * z_acc[i[0]][i[1]][i[2] + 1]) / (dz * 2)
                    + dt * (Uold_acc[i] + Uold_acc[i[0]][i[1]][i[2] - 2] * z_acc[i[0]][i[1]][i[2] - 1]) / (dz * 2));
            }
            });
        });

    deviceQ.wait();
}

template <size_t N>
void update_E(scl::queue& deviceQ, std::vector<Buffer3D>& Unew, std::vector<Buffer3D>& Uold, double dt,
    MainParams mp, std::array<Buffer3D, 4>& fq_buffer) {
    scl::range<3> num_items{ n, n, n };

    deviceQ.submit([&](scl::handler& h) {
        auto Uold_acc = Uold[4].get_access<dp_read>(h);
        auto Unew_acc = Unew[4].get_access<read_write>(h);
        auto Uro_acc = Uold[0].get_access<dp_read>(h);
        auto x_acc = Uold[1].get_access<dp_read>(h);
        auto y_acc = Uold[2].get_access<dp_read>(h);
        auto z_acc = Uold[3].get_access<dp_read>(h);
        auto q_acc = fq_buffer[3].get_access<dp_read>(h);
        auto fx_acc = fq_buffer[0].get_access<dp_read>(h);
        auto fy_acc = fq_buffer[1].get_access<dp_read>(h);
        auto fz_acc = fq_buffer[2].get_access<dp_read>(h);
        auto gamma = mp.gamma, k = mp.k, c = mp.c, dx = mp.dx, dy = mp.dy, dz = mp.dz;

        h.parallel_for<class energy_update>(num_items, [=](scl::id<3> i) {

            Unew_acc[i] = Uold_acc[i];
            // TODO likely 1/2 performance, need to fix!
            if (i[0] % 2 == 1 && i[0] >= 2 && i[0] <= (n - 3)
                && i[1] % 2 == 1 && i[1] >= 2 && i[1] <= (n - 3)
                && i[2] % 2 == 1 && i[2] >= 2 && i[2] <= (n - 3)) {
                Unew_acc[i] = (Uold_acc[i]
                    - dt * gamma * (Uold_acc[i] + Uold_acc[i[0] + 2][i[1]][i[2]]) * x_acc[i[0] + 1][i[1]][i[2]] / (dx * 2.0)
                    + dt * gamma * (Uold_acc[i] + Uold_acc[i[0] - 2][i[1]][i[2]]) * x_acc[i[0] - 1][i[1]][i[2]] / (dx * 2.0)
                    - dt * gamma * (Uold_acc[i] + Uold_acc[i[0]][i[1] + 2][i[2]]) * y_acc[i[0]][i[1] + 1][i[2]] / (dy * 2.0)
                    + dt * gamma * (Uold_acc[i] + Uold_acc[i[0]][i[1] - 2][i[2]]) * y_acc[i[0]][i[1] - 1][i[2]] / (dy * 2.0)
                    - dt * gamma * (Uold_acc[i] + Uold_acc[i[0]][i[1]][i[2] + 2]) * z_acc[i[0]][i[1]][i[2] + 1] / (dz * 2.0)
                    + dt * gamma * (Uold_acc[i] + Uold_acc[i[0]][i[1]][i[2] - 2]) * z_acc[i[0]][i[1]][i[2] - 1] / (dz * 2.0)
                    + dt * Uro_acc[i] * q_acc[i]
                    + dt * Uro_acc[i] * (fx_acc[i] * (x_acc[i[0] + 1][i[1]][i[2]] + x_acc[i[0] - 1][i[1]][i[2]]) / 2.0
                        + fy_acc[i] * (y_acc[i[0]][i[1] + 1][i[2]] + y_acc[i[0]][i[1] - 1][i[2]]) / 2.0
                        + fz_acc[i] * (z_acc[i[0]][i[1]][i[2] + 1] + z_acc[i[0]][i[1]][i[2] - 1]) / 2.0)
                        );

                Unew_acc[i] = (Unew_acc[i]
                    + dt * k * (Uold_acc[i[0] + 2][i[1]][i[2]] / (Uro_acc[i[0] + 2][i[1]][i[2]] * c) - Uold_acc[i] / (Uro_acc[i] * c)) / (dx * dx)
                    + dt * k * (Uold_acc[i[0] - 2][i[1]][i[2]] / (Uro_acc[i[0] - 2][i[1]][i[2]] * c) - Uold_acc[i] / (Uro_acc[i] * c)) / (dx * dx)
                    + dt * k * (Uold_acc[i[0]][i[1] + 2][i[2]] / (Uro_acc[i[0]][i[1] + 2][i[2]] * c) - Uold_acc[i] / (Uro_acc[i] * c)) / (dy * dy)
                    + dt * k * (Uold_acc[i[0]][i[1] - 2][i[2]] / (Uro_acc[i[0]][i[1] - 2][i[2]] * c) - Uold_acc[i] / (Uro_acc[i] * c)) / (dy * dy)
                    + dt * k * (Uold_acc[i[0]][i[1]][i[2] + 2] / (Uro_acc[i[0]][i[1]][i[2] + 2] * c) - Uold_acc[i] / (Uro_acc[i] * c)) / (dz * dz)
                    + dt * k * (Uold_acc[i[0]][i[1]][i[2] - 2] / (Uro_acc[i[0]][i[1]][i[2] - 2] * c) - Uold_acc[i] / (Uro_acc[i] * c)) / (dz * dz)
                    );
            }
            });
        });
    deviceQ.wait();
}

template <size_t N>
void update_V(scl::queue& deviceQ, std::vector<Buffer3D>& Unew, std::vector<Buffer3D>& Uold, double dt,
    MainParams mp, std::array<Buffer3D, 4>& fq_buffer) {
    scl::range<3> num_items{ n, n, n };

    deviceQ.submit([&](scl::handler& h) {
        auto ro_acc = Uold[0].get_access<dp_read>(h);
        auto e_acc = Uold[4].get_access<dp_read>(h);
        auto x_acc = Unew[1].get_access<dp_write>(h);
        auto y_acc = Unew[2].get_access<dp_write>(h);
        auto z_acc = Unew[3].get_access<dp_write>(h);
        auto xold_acc = Uold[1].get_access<dp_read>(h);
        auto yold_acc = Uold[2].get_access<dp_read>(h);
        auto zold_acc = Uold[3].get_access<dp_read>(h);
        auto fx_acc = fq_buffer[0].get_access<dp_read>(h);
        auto fy_acc = fq_buffer[1].get_access<dp_read>(h);
        auto fz_acc = fq_buffer[2].get_access<dp_read>(h);
        auto gamma = mp.gamma, dx = mp.dx, dy = mp.dy, dz = mp.dz;

        h.parallel_for<class vector_update>(num_items, [=](scl::id<3> i) {
            // TODO performance likely drops 1/2, fix in future
            if (i[0] % 2 == 1 && i[0] >= 2 && i[0] <= (n - 1)
                && i[1] % 2 == 1 && i[1] >= 2 && i[1] <= (n - 1)
                && i[2] % 2 == 1 && i[2] >= 2 && i[2] <= (n - 1)) {
                x_acc[i[0] - 1][i[1]][i[2]] = (xold_acc[i[0] - 1][i[1]][i[2]]
                    - 2 * dt * (e_acc[i] * (gamma - 1)) / (dx * (ro_acc[i] + ro_acc[i[0] - 2][i[1]][i[2]]) / 2)
                    + 2 * dt * (e_acc[i[0] - 2][i[1]][i[2]] * (gamma - 1)) / (dx * (ro_acc[i] + ro_acc[i[0] - 2][i[1]][i[2]]) / 2)
                    + dt * (fx_acc[i] + fx_acc[i[0] - 2][i[1]][i[2]]) / 2
                    );

                y_acc[i[0]][i[1] - 1][i[2]] = (yold_acc[i[0]][i[1] - 1][i[2]]
                    - 2 * dt * (e_acc[i] * (gamma - 1)) / (dy * (ro_acc[i] + ro_acc[i[0]][i[1] - 2][i[2]]) / 2)
                    + 2 * dt * (e_acc[i[0]][i[1] - 2][i[2]] * (gamma - 1)) / (dy * (ro_acc[i] + ro_acc[i[0]][i[1] - 2][i[2]]) / 2)
                    + dt * (fy_acc[i] + fy_acc[i[0]][i[1] - 2][i[2]]) / 2
                    );

                z_acc[i[0]][i[1]][i[2] - 1] = (zold_acc[i[0]][i[1]][i[2] - 1]
                    - 2 * dt * (e_acc[i] * (gamma - 1)) / (dz * (ro_acc[i] + ro_acc[i[0]][i[1]][i[2] - 2]) / 2)
                    + 2 * dt * (e_acc[i[0]][i[1]][i[2] - 2] * (gamma - 1)) / (dz * (ro_acc[i] + ro_acc[i[0]][i[1]][i[2] - 2]) / 2)
                    + dt * (fz_acc[i] + fz_acc[i[0]][i[1]][i[2] - 2]) / 2
                    );
            }
            });
        });
    deviceQ.wait();
}

template <size_t N>
void handle_walls(scl::queue& deviceQ, std::vector<Buffer3D>& U) {
    scl::range<3> num_items{ n, n, n };

    deviceQ.submit([&](scl::handler& h) {
        auto U_acc = U[0].get_access<read_write>(h);
        auto x_acc = U[1].get_access<dp_write>(h);
        auto y_acc = U[2].get_access<dp_write>(h);
        auto z_acc = U[3].get_access<dp_write>(h);

        h.parallel_for<class walls_for>(num_items, [=](scl::id<3> i) {
            U_acc[0][i[1]][i[2]] = U_acc[1][i[1]][i[2]];
            U_acc[i[0]][0][i[2]] = U_acc[i[0]][1][i[2]];
            U_acc[i[0]][i[1]][0] = U_acc[i[0]][i[1]][1];

            U_acc[n - 1][i[1]][i[2]] = U_acc[n - 2][i[1]][i[2]];
            U_acc[i[0]][n - 1][i[2]] = U_acc[i[0]][n - 2][i[2]];
            U_acc[i[0]][i[1]][n - 1] = U_acc[i[0]][i[1]][n - 2];

            x_acc[1][i[1]][i[2]] = 0;
            x_acc[n - 2][i[1]][i[2]] = 0;

            y_acc[i[0]][1][i[2]] = 0;
            y_acc[i[0]][n - 2][i[2]] = 0;

            z_acc[i[0]][i[1]][1] = 0;
            z_acc[i[0]][i[1]][n - 2] = 0;
            });
        });
    deviceQ.wait();
}

template <size_t N>
void update(scl::queue& deviceQ, MainParams& mainParams, std::vector<Buffer3D>& Unew, std::vector<Buffer3D>& Uold,
    std::array<Buffer3D, 4>& fq_buffer, scl::range<3> num_items) {
    // double dt = 0.01 * (getAbsMax(deviceQ, Uold, num_items, 1) / mainParams.dx
    //     + getAbsMax(deviceQ, Uold, num_items, 2) / mainParams.dy
    //     + getAbsMax(deviceQ, Uold, num_items, 3) / mainParams.dz
    //     + mainParams.v_sound * (1 / sqrt(pow(1 / mainParams.dx, 2) + pow(1 / mainParams.dy, 2) + pow(1 / mainParams.dz, 2))));

    double dt = 1e-7;

    update_RO<N>(deviceQ, Unew, Uold, dt, mainParams);
    update_E<N>(deviceQ, Unew, Uold, dt, mainParams, fq_buffer);
    update_V<N>(deviceQ, Unew, Uold, dt, mainParams, fq_buffer);

    handle_walls<N>(deviceQ, Unew);
}

scl::queue create_device_queue() {
    scl::default_selector dselector;

    // create an async exception handler so the program fails more gracefully.
    auto ehandler = [](scl::exception_list exceptionList) {
        for (std::exception_ptr const& e : exceptionList) {
            try {
                std::rethrow_exception(e);
            }
            catch (scl::exception const& e) {
                std::cout << "Caught an asynchronous DPC++ exception, terminating the "
                    "program."
                    << std::endl;
                std::cout << EXCEPTION_MSG;
                std::terminate();
            }
        }
    };

    try {
        // create the devices queue with the selector above and the exception
        // handler to catch async runtime errors the device queue is used to enqueue
        // the kernels and encapsulates all the states needed for execution
        scl::queue q(dselector, ehandler);

        return q;
    }
    catch (scl::exception const& e) {
        // catch the exception from devices that are not supported.
        std::cout << "An exception is caught when creating a device queue."
            << std::endl;
        std::cout << EXCEPTION_MSG;
        std::terminate();
    }
}

int main() {
    scl::queue deviceQ = create_device_queue();
    std::cout << "Device: " << deviceQ.get_device().get_info<scl::info::device::name>() << std::endl;

    // Initialize structures
    auto mainParams = MainParams{};
    static Tensor3D<double> q{0}, fx{0}, fy{0}, fz{0};
    static Tensor4D<double, params> U, corrector{}, predictor{};
    scl::range<3> num_items{ n, n, n };
    fillTensor4D<double, params>(U, std::array<double, params> {1.25, 0.0, 0.0, 0.0, 300.0 * mainParams.c * 1.25 * 1.25});

    auto tempp = static_cast<size_t>(n / 2 + 1);
    std::array<double, params> initial_U{ 1.25, 0.0, 0.0, 0.0, 400.0 * mainParams.c * 1.25 * 1.25};

    for (size_t param = 0; param < params; param++) {
        U[param][tempp * n * n + tempp * n + tempp] = initial_U[param];
    }

    // Initialize buffers
    // fx, fy, fz, q
    std::array<Buffer3D, 4> fq_buffers{ Buffer3D{fx.data(), num_items}, Buffer3D{fy.data(), num_items},
        Buffer3D{fz.data(), num_items}, Buffer3D{q.data(), num_items} };
    // U, corrector, predictor
    static std::array<std::vector<Buffer3D>, 3> buffers_4d;
    // max x, y, z
    //scl::buffer<double, 1> max_v(std::array<double, 3>{0, 0, 0}.data(), num_max);

    for (size_t i = 0; i < params; i++) {
        buffers_4d[0].emplace_back(U[i].data(), num_items);
        buffers_4d[1].emplace_back(corrector[i].data(), num_items);
        buffers_4d[2].emplace_back(predictor[i].data(), num_items);
    }

    // Main loop

    auto begin = std::chrono::system_clock::now();

    for (size_t t = 0; t < total_time; t++) {
        update<params>(deviceQ, mainParams, buffers_4d[2], buffers_4d[0], fq_buffers, num_items);
        update<params>(deviceQ, mainParams, buffers_4d[1], buffers_4d[2], fq_buffers, num_items);
        add4DTensorsScaled<params>(deviceQ, buffers_4d[0], buffers_4d[1], buffers_4d[2], num_items);
    }

    auto end = std::chrono::system_clock::now();
    std::cout << double((end - begin).count()) << std::endl;

    return 0;
}
