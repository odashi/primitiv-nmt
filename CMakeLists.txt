cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

project(primitiv-nmt VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(PRIMITIV_NMT_USE_CUDA "Whether or not to use CUDA." OFF)

find_package(Protobuf REQUIRED)
find_package(Primitiv REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Werror -fPIC")
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_BINARY_DIR}
  ${PROTOBUF_INCLUDE_DIR}
  ${PRIMITIV_INCLUDE_DIR}
)

add_subdirectory(primitiv_nmt)
