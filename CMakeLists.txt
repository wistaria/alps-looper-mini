#  Copyright Haruhiko Matsuo and Synsge Todo 2011.
#  Distributed under the Boost Software License, Version 1.0.
#      (See http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(alps-looper-standalone)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/config")

#
# Options
#
option(LOOPER_ENABLE_OPENMP "OpenMP is enabled by default. You can turn off OpenMP by -DLOOPER_ENABLE_OPENMP=OFF" ON)

option(LOOPER_TIMER "You can turn on Timer by -DLOOPER_TIMER=ON" OFF)
if(LOOPER_TIMER)
  add_definitions(-DALPS_ENABLE_TIMER)
endif(LOOPER_TIMER)

if(LOOPER_TIMER)
  option(LOOPER_TIMER_TRACE "You can turn on debug trace output in timer by -DLOOPER_TIMER_TRACE=ON" OFF)
  if(LOOPER_TIMER_TRACE)
    add_definitions(-DALPS_ENABLE_TIMER_TRACE)
  endif(LOOPER_TIMER_TRACE)

  option(LOOPER_TIMER_DETAILED "You can turn on detailed timer report by -DLOOPER_TIMER_DETAILED=ON" OFF)
  if(LOOPER_TIMER_DETAILED)
    add_definitions(-DALPS_ENABLE_TIMER_DETAILED)
  endif(LOOPER_TIMER_DETAILED)

  option(LOOPER_TIMER_BARRIER "You can turn on barrier synchronization in timer by -DLOOPER_TIMER_BARRIER=ON" OFF)
  if(LOOPER_TIMER_BARRIER)
    add_definitions(-DALPS_ENABLE_TIMER_BARRIER)
  endif(LOOPER_TIMER_BARRIER)

  option(LOOPER_ENABLE_FIPP "Turn on fipp by -DLOOPER_ENABLE_FIPP=ON" OFF)
  option(LOOPER_ENABLE_FAPP "Turn on fapp by -DLOOPER_ENABLE_FAPP=ON" OFF)
  option(LOOPER_ENABLE_FAPP_PA "Turn on fapp PA by -DLOOPER_ENABLE_FAPP_PA=ON" OFF)
  if (LOOPER_ENABLE_FIPP OR LOOPER_ENABLE_FAPP OR LOOPER_ENABLE_FAPP_PA)
    set(CMAKE_SKIP_RPATH True)
  endif(LOOPER_ENABLE_FIPP OR LOOPER_ENABLE_FAPP OR LOOPER_ENABLE_FAPP_PA)
  if(LOOPER_ENABLE_FIPP)
    add_definitions(-DALPS_ENABLE_FIPP)
  endif(LOOPER_ENABLE_FIPP)
  if(LOOPER_ENABLE_FAPP)
    add_definitions(-DALPS_ENABLE_FAPP)
  endif(LOOPER_ENABLE_FAPP)
  if(LOOPER_ENABLE_FAPP_PA)
    add_definitions(-DALPS_ENABLE_FAPP_PA)
  endif(LOOPER_ENABLE_FAPP_PA)
  if(LOOPER_ENABLE_FIPP AND LOOPER_ENABLE_FAPP)
    message(FATAL_ERROR "Both of sampler and hardware counter can't be enabled.")
  endif(LOOPER_ENABLE_FIPP AND LOOPER_ENABLE_FAPP) 
  if(LOOPER_ENABLE_FIPP AND LOOPER_ENABLE_FAPP_PA)
    message(FATAL_ERROR "Both of sampler and hardware counter can't be enabled.")
  endif(LOOPER_ENABLE_FIPP AND LOOPER_ENABLE_FAPP_PA) 
  if(LOOPER_ENABLE_FAPP AND LOOPER_ENABLE_FAPP_PA) 
    message(FATAL_ERROR "Both of sampler and hardware counter can't be enabled.")
  endif(LOOPER_ENABLE_FAPP AND LOOPER_ENABLE_FAPP_PA) 
endif(LOOPER_TIMER)

option(LOOPER_USE_DETERMINISTIC_UNIFY "Use deterministic unify for debug" OFF)
if(LOOPER_USE_DETERMINISTIC_UNIFY)
  add_definitions(-DLOOPER_USE_DETERMINISTIC_UNIFY)
endif(LOOPER_USE_DETERMINISTIC_UNIFY)

if(NOT LOOPER_FRAGMENT_ORDER)
  set(LOOPER_FRAGMENT_ORDER 0)
endif(NOT LOOPER_FRAGMENT_ORDER)
add_definitions(-DLOOPER_FRAGMENT_ORDER=${LOOPER_FRAGMENT_ORDER})

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
list(APPEND TARGETS loop loop0)
list(APPEND TARGETS loop_u) 
list(APPEND TARGETS bond_percolation)
list(APPEND TARGETS sw sw_r)
list(APPEND TARGETS visualize_graph vmpeak)

foreach(name ${TARGETS})
  if(EXISTS ${PROJECT_SOURCE_DIR}/${name}.C)
    add_executable(${name} ${name}.C)
    add_alps_test(${name})
  endif(EXISTS ${PROJECT_SOURCE_DIR}/${name}.C)
endforeach(name TARGETS)

if(MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})

  list(APPEND TARGETS_MPI loop_mpi)
  list(APPEND TARGETS_MPI loop_mpi_u)

  foreach(name ${TARGETS_MPI})
    if(EXISTS ${PROJECT_SOURCE_DIR}/${name}.C)
      add_executable(${name} ${name}.C)
      target_link_libraries(${name} ${MPI_LIBRARIES})
      foreach(np RANGE 1 4)
        add_alps_test_mpi(${name} ${np})
      endforeach(np)
    endif(EXISTS ${PROJECT_SOURCE_DIR}/${name}.C)
  endforeach(name)

  set(TARGETS_MPI)
  list(APPEND TARGETS_MPI communication_test)
  list(APPEND TARGETS_MPI parallel_gen_random)
  list(APPEND TARGETS_MPI parallel_gen_random_fmpi)
  list(APPEND TARGETS_MPI parallel_push_back_const)
  list(APPEND TARGETS_MPI parallel_push_back_random)
  list(APPEND TARGETS_MPI parallel_check_nanosleep)
  list(APPEND TARGETS_MPI parallel_fill_times_kernel)
  list(APPEND TARGETS_MPI parallel_ir_operators_kernel)
  list(APPEND TARGETS_MPI parallel_unify_test)

  foreach(name ${TARGETS_MPI})
    if(EXISTS ${PROJECT_SOURCE_DIR}/${name}.C)
      add_executable(${name} ${name}.C)
      target_link_libraries(${name} ${MPI_LIBRARIES})
    endif(EXISTS ${PROJECT_SOURCE_DIR}/${name}.C)
  endforeach(name)
endif(MPI_FOUND)
