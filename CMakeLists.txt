cmake_minimum_required(VERSION 3.16)
project(MPI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

option(MPI_ACTIVE OFF)

if (MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif ()

set(Boost_USE_STATIC_LIBS OFF) # enable dynamic linking
set(Boost_USE_MULTITHREAD ON)  # enable multithreading
find_package(TBB REQUIRED)
find_package(ArrayFire REQUIRED)
find_package(Threads REQUIRED)
if (MPI_ACTIVE)
    find_package(MPI REQUIRED)
    find_package(Boost REQUIRED COMPONENTS mpi serialization)
else()
    find_package(Boost REQUIRED COMPONENTS serialization)
endif()

add_executable(mpi_app src/mpi.cpp)
add_executable(array_app src/arrayf.cpp)


if(Boost_FOUND)
    target_link_libraries(mpi_app ${Boost_LIBRARIES} ArrayFire::af Threads::Threads TBB::tbb)
    target_link_libraries(array_app ${Boost_LIBRARIES} ArrayFire::af Threads::Threads TBB::tbb)
    target_include_directories(mpi_app PRIVATE ${Boost_INCLUDE_DIRS} ${MPI_INCLUDE_PATH} ${ArrayFire_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS} include)
    target_include_directories(array_app PRIVATE ${Boost_INCLUDE_DIRS} ${MPI_INCLUDE_PATH} ${ArrayFire_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS} include)
endif()
