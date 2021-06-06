#define ONEDPL_USE_DPCPP_BACKEND 1
#define ONEDPL_USE_TBB_BACKEND 0
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>
#if defined(FPGA) || defined(FPGA_EMULATOR)
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif
#include <array>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <exception>

// output message for runtime exceptions
#define EXCEPTION_MSG \
  "    If you are targeting an FPGA hardware, please ensure that an FPGA board is plugged to the system, \n\
        set up correctly and compile with -DFPGA  \n\
    If you are targeting the FPGA emulator, compile with -DFPGA_EMULATOR.\n"

namespace scl = cl::sycl;

struct MainParams {
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

template <typename T>
class DTensor3D {
private:
    size_t n, totalN;
    T* dataArr;
public:

    DTensor3D(size_t n_) : totalN(n_* n_* n_), dataArr(new T[totalN]), n(n_) {}

    T& operator()(size_t i, size_t ii, size_t iii) {
        if (!(i < n && ii < n && iii < n)) {
            throw std::out_of_range("indices out of range");
        }
        return dataArr[i * n * n + ii * n + iii];
    }

    const T& operator()(size_t i, size_t ii, size_t iii) const {
        if (!(i < n && ii < n && iii < n)) {
            throw std::out_of_range("indices out of range");
        }
        return dataArr[i * n * n + ii * n + iii];
    }

    void fill(T value) noexcept {
        for (size_t i = 0; i < totalN; i++) {
            dataArr[i] = value;
        }
    }

    T* data() noexcept {
        return dataArr;
    }

    const T* data() const noexcept {
        return dataArr;
    }

    ~DTensor3D() {
        delete[] dataArr;
    }
};

using Buffer3D = scl::buffer<double, 3>;

// Convience data access definitions
constexpr scl::access::mode dp_read = scl::access::mode::read;
constexpr scl::access::mode dp_write = scl::access::mode::write;
constexpr scl::access::mode read_write = scl::access::mode::read_write;

// key parmeters
constexpr size_t params = 5;

template <typename T, size_t N> using DTensor4D = std::array<DTensor3D<T>, N>;

template <size_t N>
void add4DTensorsScaled(scl::queue& deviceQ, std::vector<Buffer3D>& result, std::vector<Buffer3D>& lhs, std::vector<Buffer3D>& rhs, scl::range<3>& num_items, double scale = 0.5) {
    for (size_t param = 0; param < N; param++) {
        deviceQ.submit([&](scl::handler& h) {
            auto lhs_acc = lhs[param].get_access<dp_read>(h);
            auto rhs_acc = rhs[param].get_access<dp_read>(h);

            auto sum_acc = result[param].get_access<dp_write, scl::access::target::global_buffer>(h);

            h.parallel_for<class vector_addition>(num_items, [=](scl::id<3> i) {
                sum_acc[i] = scale * (lhs_acc[i] + rhs_acc[i]);
                });
            });
    }
}

double getAbsMax(scl::queue& deviceQ, std::vector<Buffer3D>& tensor, scl::range<3> num_items, size_t last_id) {
    auto policy = oneapi::dpl::execution::make_device_policy(deviceQ);
    sycl::range<1> abs_range{ num_items.size() };
    auto tempBuf = tensor[last_id].reinterpret<double, 1>(abs_range);

    return oneapi::dpl::reduce(policy, oneapi::dpl::begin(tempBuf),
        oneapi::dpl::end(tempBuf), 0, oneapi::dpl::maximum<double>{});
}

template <size_t N>
void update_RO(scl::queue& deviceQ, std::vector<Buffer3D>& Unew, std::vector<Buffer3D>& Uold, double dt, MainParams mp, scl::range<3> num_items) {

    deviceQ.submit([&](scl::handler& h) {
        auto Uold_acc = Uold[0].get_access<dp_read>(h);
        auto Unew_acc = Unew[0].get_access<dp_write>(h);
        auto x_acc = Uold[1].get_access<dp_read>(h);
        auto y_acc = Uold[2].get_access<dp_read>(h);
        auto z_acc = Uold[3].get_access<dp_read>(h);
        auto dx = mp.dx, dy = mp.dy, dz = mp.dz;

        h.parallel_for<class ro_update>(num_items, [=](scl::id<3> i) {
            size_t n = num_items[0];
            Unew_acc[i] = Uold_acc[i];
             //TODO likely 1/2 performance, need to fix!
            if (i[0] % 2 == 1 && i[0] >= 2 && i[0] < (n - 2)
                && i[1] % 2 == 1 && i[1] >= 2 && i[1] < (n - 2)
                && i[2] % 2 == 1 && i[2] >= 2 && i[2] < (n - 2)) {
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

}

template <size_t N>
void update_E(scl::queue& deviceQ, std::vector<Buffer3D>& Unew, std::vector<Buffer3D>& Uold, double dt,
    MainParams mp, std::array<Buffer3D, 4>& fq_buffer, scl::range<3> num_items) {

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
            size_t n = num_items[0];
            Unew_acc[i] = Uold_acc[i];
            // TODO likely 1/2 performance, need to fix!
            if (i[0] % 2 == 1 && i[0] >= 2 && i[0] < (n - 2)
                && i[1] % 2 == 1 && i[1] >= 2 && i[1] < (n - 2)
                && i[2] % 2 == 1 && i[2] >= 2 && i[2] < (n - 2)) {
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
}

template <size_t N>
void update_V(scl::queue& deviceQ, std::vector<Buffer3D>& Unew, std::vector<Buffer3D>& Uold, double dt,
    MainParams mp, std::array<Buffer3D, 4>& fq_buffer, scl::range<3> num_items) {

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
            size_t n = num_items[0];
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
}

template <size_t N>
void handle_walls(scl::queue& deviceQ, std::vector<Buffer3D>& U, scl::range<3> num_items) {

    deviceQ.submit([&](scl::handler& h) {
        auto U_acc = U[0].get_access<read_write>(h);
        auto x_acc = U[1].get_access<dp_write>(h);
        auto y_acc = U[2].get_access<dp_write>(h);
        auto z_acc = U[3].get_access<dp_write>(h);

        h.parallel_for<class walls_for>(num_items, [=](scl::id<3> i) {
            size_t n = num_items[0];
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
}

template <size_t N>
void update(scl::queue& deviceQ, MainParams& mainParams, std::vector<Buffer3D>& Unew, std::vector<Buffer3D>& Uold,
    std::array<Buffer3D, 4>& fq_buffer, scl::range<3> num_items) {
    // disable if library unavailable
   double dt = 0.02 * (getAbsMax(deviceQ, Uold, num_items, 1) / mainParams.dx
        + getAbsMax(deviceQ, Uold, num_items, 2) / mainParams.dy
        + getAbsMax(deviceQ, Uold, num_items, 3) / mainParams.dz
        + mainParams.v_sound * (1 / sqrt(pow(1 / mainParams.dx, 2) + pow(1 / mainParams.dy, 2) + pow(1 / mainParams.dz, 2))));

    update_RO<N>(deviceQ, Unew, Uold, dt, mainParams, num_items);
    update_E<N>(deviceQ, Unew, Uold, dt, mainParams, fq_buffer, num_items);
    update_V<N>(deviceQ, Unew, Uold, dt, mainParams, fq_buffer, num_items);

    handle_walls<N>(deviceQ, Unew, num_items);
    deviceQ.wait_and_throw();
}

//************************************
// Function description: create a device queue with the default selector or
// explicit FPGA selector when FPGA macro is defined
//    return: DPC++ queue object
//************************************
sycl::queue create_device_queue() {
    // create device selector for the device of your interest
#ifdef FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card
    INTEL::fpga_emulator_selector dselector;
#elif defined(FPGA)
  // DPC++ extension: FPGA selector on systems with FPGA card
    INTEL::fpga_selector dselector;
#else
  // the default device selector: it will select the most performant device
  // available at runtime.
    bool foundGPU = false;
    sycl::device selectedDevice;
    for (auto& device : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (device.get_info<sycl::info::device::name>() == "Intel(R) UHD Graphics 620") {
            selectedDevice = device;
            foundGPU = true;
        }
    }
    //if (!foundGPU) {
        selectedDevice = sycl::cpu_selector{}.select_device();
    //}

#endif

    /*auto wgroup_size = selectedDevice.get_info<sycl::info::device::max_work_group_size>();
    if (wgroup_size % 2 != 0) {
        throw "Work-group size has to be even!";
    }

    auto has_local_mem = selectedDevice.is_host()
        || (selectedDevice.get_info<sycl::info::device::local_mem_type>()
            != sycl::info::local_mem_type::none);
    auto local_mem_size = selectedDevice.get_info<sycl::info::device::local_mem_size>();
    if (!has_local_mem
        || local_mem_size < (wgroup_size * sizeof(int32_t)))
    {
        throw "Device doesn't have enough local memory!";
    }*/

    // create an async exception handler so the program fails more gracefully.
    auto ehandler = [](cl::sycl::exception_list exceptionList) {
        for (std::exception_ptr const& e : exceptionList) {
            try {
                std::rethrow_exception(e);
            }
            catch (cl::sycl::exception const& e) {
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
        sycl::queue q(selectedDevice, ehandler);

        return q;
    }
    catch (cl::sycl::exception const& e) {
        // catch the exception from devices that are not supported.
        std::cout << "An exception is caught when creating a device queue."
            << std::endl;
        std::cout << EXCEPTION_MSG;
        std::terminate();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cout << "DOUN" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    int timeN = std::stoi(argv[2]);
    int logfreq = std::stoi(argv[3]);
    int savefreq = std::stoi(argv[4]);

    // Set constants
    scl::queue deviceQ = create_device_queue();

    std::cout << "Device: " << deviceQ.get_device().get_info<scl::info::device::name>() << std::endl;

    std::ofstream csv_file;

    // Initialize structures
    auto mainParams = MainParams{};
    DTensor3D<double> q{n}, fx{n}, fy{n}, fz{n};
    DTensor4D<double, params> U{n,n,n,n,n}, corrector{ n,n,n,n,n }, predictor{ n,n,n,n,n };
    scl::range<3> num_items{ n, n, n };
    {
        std::array<double, params> values{ 1.25, 0.0, 0.0, 0.0, 300.0 };
        for (size_t iv = 0; iv < params; iv++) {
            U[iv].fill(values[iv]);
        }

        // initial point
        auto tempp = static_cast<size_t>(n / 2 + 1);
        std::array<double, params> initial_U{ 1.25, 0.0, 0.0, 0.0, 400.0 };

        for (size_t param = 0; param < params; param++) {
            U[param](tempp, tempp, tempp) = initial_U[param];
        }
    }

    // Initialize buffers
    // fx, fy, fz, q
    std::array<Buffer3D, 4> fq_buffers{ Buffer3D{fx.data(), num_items}, Buffer3D{fy.data(), num_items},
        Buffer3D{fz.data(), num_items}, Buffer3D{q.data(), num_items} };
    // U, corrector, predictor
    std::array<std::vector<Buffer3D>, 3> buffers_4d;
    // max x, y, z
    //scl::buffer<double, 1> max_v(std::array<double, 3>{0, 0, 0}.data(), num_max);

    for (size_t i = 0; i < params; i++) {
        buffers_4d[0].emplace_back(U[i].data(), num_items);
        buffers_4d[1].emplace_back(corrector[i].data(), num_items);
        buffers_4d[2].emplace_back(predictor[i].data(), num_items);
    }

    // Main loop

    //csv_file.open("data.csv");
    for (size_t t = 0; t < timeN; t++) {
        update<params>(deviceQ, mainParams, buffers_4d[2], buffers_4d[0], fq_buffers, num_items);

        if (t % logfreq == 0) {
            std::cout << "Time: " << t << std::endl;
        }

        //if (t % savefreq == 0) {
        //    //auto e_acc = buffers_4d[2][4].get_access<dp_read>();
        //    //auto vx_acc = buffers_4d[2][1].get_access<dp_read>();
        //    //auto vy_acc = buffers_4d[2][2].get_access<dp_read>();
        //    //auto vz_acc = buffers_4d[2][3].get_access<dp_read>();

        //    //for (size_t i = 0; i < n; i++) {
        //    //    for (size_t ii = 0; ii < n; ii++) {
        //    //        for (size_t iii = 0; iii < n; iii++) {
        //    //            csv_file << i << ',' << ii << ',' << iii << ','
        //    //                << vx_acc[i][ii][iii] << ',' << vy_acc[i][ii][iii] << ',' << vz_acc[i][ii][iii] << ','
        //    //                << e_acc[i][ii][iii] << '\n';
        //    //        }
        //    //    }
        //    //}
        //    //csv_file << '\n';
        //}

        update<params>(deviceQ, mainParams, buffers_4d[1], buffers_4d[2], fq_buffers, num_items);
        add4DTensorsScaled<params>(deviceQ, buffers_4d[0], buffers_4d[1], buffers_4d[2], num_items);
    }

    return 0;
}
