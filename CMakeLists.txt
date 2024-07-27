cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(FETCHCONTENT_QUIET OFF CACHE BOOL "" FORCE)

include(cmake/cpm.cmake)

project(kn-benchmark)

add_executable(kn-benchmark src/main.cpp)
