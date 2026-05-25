cmake_minimum_required(VERSION 3.20)
project(my_Logger)
set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_compile_options(/utf-8)
endif()
add_executable(MY_LOGGER test_main.cpp)