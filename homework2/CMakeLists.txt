cmake_minimum_required (VERSION 2.6)
project (hw2)

set(CMAKE_CXX_FLAGS "-std=c++0x -Wall -Werror -g -ggdb -O0 -O2 -pg")
#set(CMAKE_CXX_FLAGS "-std=c++0x -O0 -O2")

include_directories(${src})
set(SOURCES src/main.cpp)

add_executable(homework2 src/hw2.cpp ${SOURCES})
