# CMakeList.txt: проект CMake для beGAN; 
cmake_minimum_required (VERSION 3.8)

project(beGAN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.0 REQUIRED)
#find_package(DLIB REQUIRED)

include("CMakeLists.Files.txt")

include_directories(${CMAKE_SOURCE_DIR}/src)
#include_directories(${DLIB_INCLUDE_DIRECTORY})

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (MSVC) 
add_definitions(-D_DISABLE_EXTENDED_ALIGNED_STORAGE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()
add_definitions(-DUSE_MXNET)
add_definitions(-DDLIB_USE_CUDA)		#for dlib
add_definitions(-DDLIB_HAVE_SSE2)		#for dlib

source_group("Headers" FILES ${HEADERS})
set(SOURCES ${SOURCES} ${HEADERS})

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${LIBS}) #dlib::dlib)