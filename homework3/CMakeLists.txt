cmake_minimum_required (VERSION 2.6)
project (hw3)

set(CMAKE_CXX_FLAGS "-std=c++0x -Wall -Werror -g -ggdb -O0 -O2")
#set(CMAKE_CXX_FLAGS "-std=c++0x -O0 -O2")

include_directories(${src})
set(SOURCES src/hw3.cpp)

add_executable(homework3 src/main.cpp ${SOURCES})
