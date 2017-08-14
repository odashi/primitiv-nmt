cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

project(mymt VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Protobuf REQUIRED)
find_package(Primitiv REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Werror -fPIC")
endif()

add_subdirectory(mymt)
