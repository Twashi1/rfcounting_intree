cmake_minimum_required(VERSION 3.20)
project(AddSimCommandsPass LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM 21.1.8 REQUIRED CONFIG)

message(STATUS "Using LLVM_DIR=${LLVM_DIR}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()


include(AddLLVM)

add_llvm_pass_plugin(AddSimCommandsPass
  lib/SimCommandsPass.cpp
)

target_include_directories(AddSimCommandsPass
  PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# include_directories(${LLVM_INCLUDE_DIRS})
# add_definitions(${LLVM_DEFINITIONS})

# add_library(AddSimCommandsPass MODULE lib/SimCommandsPass.cpp include/SimCommandsPass.h)

set_target_properties(AddSimCommandsPass PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  POSITION_INDEPENDENT_CODE ON
  COMPILE_FLAGS "-fno-rtti"
)
