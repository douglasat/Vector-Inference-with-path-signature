cmake_minimum_required(VERSION 3.12)
project(geometric_plan_c)

find_package(pybind11 REQUIRED)
include_directories("/usr/local/include/ompl-1.6")
include_directories("/usr/include/eigen3")

pybind11_add_module(geometric_plan_c geometric_plan.cpp)
