# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.0.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.0.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/cmosnick07/Code/hpc-work/homework0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/cmosnick07/Code/hpc-work/homework0/build

# Include any dependencies generated for this target.
include CMakeFiles/../homework0.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/../homework0.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/../homework0.dir/flags.make

CMakeFiles/../homework0.dir/main.cpp.o: CMakeFiles/../homework0.dir/flags.make
CMakeFiles/../homework0.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/cmosnick07/Code/hpc-work/homework0/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/../homework0.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/../homework0.dir/main.cpp.o -c /Users/cmosnick07/Code/hpc-work/homework0/main.cpp

CMakeFiles/../homework0.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/../homework0.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/cmosnick07/Code/hpc-work/homework0/main.cpp > CMakeFiles/../homework0.dir/main.cpp.i

CMakeFiles/../homework0.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/../homework0.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/cmosnick07/Code/hpc-work/homework0/main.cpp -o CMakeFiles/../homework0.dir/main.cpp.s

CMakeFiles/../homework0.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/../homework0.dir/main.cpp.o.requires

CMakeFiles/../homework0.dir/main.cpp.o.provides: CMakeFiles/../homework0.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/../homework0.dir/build.make CMakeFiles/../homework0.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/../homework0.dir/main.cpp.o.provides

CMakeFiles/../homework0.dir/main.cpp.o.provides.build: CMakeFiles/../homework0.dir/main.cpp.o

# Object files for target ../homework0
__/homework0_OBJECTS = \
"CMakeFiles/../homework0.dir/main.cpp.o"

# External object files for target ../homework0
__/homework0_EXTERNAL_OBJECTS =

../homework0: CMakeFiles/../homework0.dir/main.cpp.o
../homework0: CMakeFiles/../homework0.dir/build.make
../homework0: CMakeFiles/../homework0.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../homework0"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/../homework0.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/../homework0.dir/build: ../homework0
.PHONY : CMakeFiles/../homework0.dir/build

CMakeFiles/../homework0.dir/requires: CMakeFiles/../homework0.dir/main.cpp.o.requires
.PHONY : CMakeFiles/../homework0.dir/requires

CMakeFiles/../homework0.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/../homework0.dir/cmake_clean.cmake
.PHONY : CMakeFiles/../homework0.dir/clean

CMakeFiles/../homework0.dir/depend:
	cd /Users/cmosnick07/Code/hpc-work/homework0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/cmosnick07/Code/hpc-work/homework0 /Users/cmosnick07/Code/hpc-work/homework0 /Users/cmosnick07/Code/hpc-work/homework0/build /Users/cmosnick07/Code/hpc-work/homework0/build /Users/cmosnick07/Code/hpc-work/homework0/build/homework0.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/../homework0.dir/depend

