cmake_minimum_required(VERSION 3.6)
project(lab1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
#include_directories(/Ubuntu/usr/lib/x86_64-linux-gnu/openmpi/include)
#include_directories(/usr/lib/x86_64-linux-gnu/openmpi/lib)
add_executable(lab1 main.cpp)
target_link_libraries(lab1 ${MPI_CXX_LIBRARIES})



target_sources(lab1 PRIVATE
    load.cpp operations.cpp
)
target_include_directories(lab1 PRIVATE
    include
)
target_compile_features(lab1 PRIVATE cxx_std_17)


