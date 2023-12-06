# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /sw/eb/sw/CMake/3.12.1-GCCcore-7.3.0/bin/cmake

# The command to remove a file.
RM = /sw/eb/sw/CMake/3.12.1-GCCcore-7.3.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chrisanand/please/cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chrisanand/please/cuda

# Include any dependencies generated for this target.
include CMakeFiles/bitonic.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bitonic.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bitonic.dir/flags.make

CMakeFiles/bitonic.dir/bitonic_sort.cu.o: CMakeFiles/bitonic.dir/flags.make
CMakeFiles/bitonic.dir/bitonic_sort.cu.o: bitonic_sort.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chrisanand/please/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/bitonic.dir/bitonic_sort.cu.o"
	/sw/eb/sw/CUDA/9.2.88/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/chrisanand/please/cuda/bitonic_sort.cu -o CMakeFiles/bitonic.dir/bitonic_sort.cu.o

CMakeFiles/bitonic.dir/bitonic_sort.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/bitonic.dir/bitonic_sort.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/bitonic.dir/bitonic_sort.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/bitonic.dir/bitonic_sort.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target bitonic
bitonic_OBJECTS = \
"CMakeFiles/bitonic.dir/bitonic_sort.cu.o"

# External object files for target bitonic
bitonic_EXTERNAL_OBJECTS =

CMakeFiles/bitonic.dir/cmake_device_link.o: CMakeFiles/bitonic.dir/bitonic_sort.cu.o
CMakeFiles/bitonic.dir/cmake_device_link.o: CMakeFiles/bitonic.dir/build.make
CMakeFiles/bitonic.dir/cmake_device_link.o: /scratch/group/csce435-f23/Caliper-CUDA/caliper/lib64/libcaliper.so.2.10.0
CMakeFiles/bitonic.dir/cmake_device_link.o: CMakeFiles/bitonic.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chrisanand/please/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/bitonic.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bitonic.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bitonic.dir/build: CMakeFiles/bitonic.dir/cmake_device_link.o

.PHONY : CMakeFiles/bitonic.dir/build

# Object files for target bitonic
bitonic_OBJECTS = \
"CMakeFiles/bitonic.dir/bitonic_sort.cu.o"

# External object files for target bitonic
bitonic_EXTERNAL_OBJECTS =

bitonic: CMakeFiles/bitonic.dir/bitonic_sort.cu.o
bitonic: CMakeFiles/bitonic.dir/build.make
bitonic: /scratch/group/csce435-f23/Caliper-CUDA/caliper/lib64/libcaliper.so.2.10.0
bitonic: CMakeFiles/bitonic.dir/cmake_device_link.o
bitonic: CMakeFiles/bitonic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chrisanand/please/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable bitonic"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bitonic.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bitonic.dir/build: bitonic

.PHONY : CMakeFiles/bitonic.dir/build

CMakeFiles/bitonic.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bitonic.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bitonic.dir/clean

CMakeFiles/bitonic.dir/depend:
	cd /home/chrisanand/please/cuda && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chrisanand/please/cuda /home/chrisanand/please/cuda /home/chrisanand/please/cuda /home/chrisanand/please/cuda /home/chrisanand/please/cuda/CMakeFiles/bitonic.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bitonic.dir/depend

