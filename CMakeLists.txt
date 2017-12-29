cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 11)
project(gru_validation)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -std=c++11 -O3")
include_directories(include external/rapidjson/include)
add_executable(gru_forward_validation src/test_gru_forward.cpp src/utils.cpp)
add_executable(gru_backward_validation src/test_gru_backward.cpp src/utils.cpp)

