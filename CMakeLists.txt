#  Copyright Haruhiko Matsuo and Synsge Todo 2011-2014.
#  Distributed under the Boost Software License, Version 1.0.
#      (See http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(alps-looper-mini NONE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/config")

#
# Options
#
option(LOOPER_ENABLE_OPENMP "OpenMP is enabled by default. You can turn off OpenMP by -DLOOPER_ENABLE_OPENMP=OFF" ON)

option(LOOPER_TIMER "Timer is enabled by defualt. You can turn off Timer by -DLOOPER_TIMER=OFF" ON)
if(LOOPER_TIMER)
  add_definitions(-DALPS_ENABLE_TIMER)
endif(LOOPER_TIMER)

#
# C++ and C compilers
#
enable_language(CXX)
enable_language(C)

#
# Required external sources/libraries.
#
## Boost srouce
find_package(BoostSrc)
include_directories(${Boost_ROOT_DIR})

## MPI
find_package(MPI)
if(NOT MPI_FOUND)
  message(WARNING "MPI was not found")
endif(NOT MPI_FOUND)

## OpenMP
if(LOOPER_ENABLE_OPENMP)
  find_package(OpenMP)
  if(NOT OPENMP_FOUND)
    message(FATAL_ERROR "OpenMP was not found")
  endif(NOT OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  add_definitions(-DLOOPER_ENABLE_OPENMP)
  message(STATUS "Enabled OpenMP")
else(LOOPER_ENABLE_OPENMP)
  message(STATUS "Disabled OpenMP")
endif(LOOPER_ENABLE_OPENMP)

## maprof
if(LOOPER_TIMER)
  set(MAPROF_SOURCES ma_prof/src/maprof.c ma_prof/src/maprof_time.c ma_prof/src/maprof_yaml.c ma_prof/src/maprof_proc.c)
  add_definitions(-DMAPROF_MAX_SECTIONS=20 -DMAPROF_OUTPUT="maprof_output.yaml")
else(LOOPER_TIMER)
  set(MAPROF_SOURCES)
endif(LOOPER_TIMER)

#
# RPATH setting
#
if(APPLE)
  set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
else(APPLE)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif(APPLE)

#
# tests
#
include(config/add_alps_test.cmake)

#
# targets
#
add_executable(loop loop.C ${MAPROF_SOURCES})
add_alps_test(loop)

if(MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})
  add_executable(loop_mpi loop_mpi.C ${MAPROF_SOURCES})
  target_link_libraries(loop_mpi ${MPI_LIBRARIES})
  foreach(np RANGE 1 4)
    add_alps_test_mpi(loop_mpi ${np})
  endforeach(np)
endif(MPI_FOUND)
