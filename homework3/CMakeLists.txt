cmake_minimum_required (VERSION 2.6)
project (hw3)

set(CMAKE_CXX_FLAGS "-std=c++0x -Wall -Werror -g -ggdb")
#set(CMAKE_CXX_FLAGS "-std=c++0x -O0 -O2")

FIND_PACKAGE( Boost 1.60 COMPONENTS thread system filesystem REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})

include_directories(${src})
set(SOURCES src/hw3.cpp src/MosnickThread.cpp)

add_executable(homework3 src/main.cpp ${SOURCES})

target_link_libraries(homework3 
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_THREAD_LIBRARY}
)