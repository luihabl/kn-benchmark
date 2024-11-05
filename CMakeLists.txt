cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
set(FETCHCONTENT_QUIET OFF CACHE BOOL "" FORCE)

# set(CMAKE_CXX_FLAGS_DEBUG "-O3")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include(cmake/cpm.cmake)

add_subdirectory(kn)

project(kn-benchmark)

CPMAddPackage("gh:d99kris/rapidcsv@8.83")

add_executable(kn-benchmark src/main.cpp)

target_link_libraries(kn-benchmark PUBLIC kn::kn rapidcsv)