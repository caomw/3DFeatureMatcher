cmake_minimum_required(VERSION 2.8)
project(3dfeaturematcher)

cmake_policy(SET CMP0015 NEW)

set(CMAKE_BUILD_TYPE "Debug")

message("Project ${PROJECT_NAME} build Type: ${CMAKE_BUILD_TYPE}")

#FIND_PACKAGE( Boost REQUIRED )
#INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIR})
link_directories ( ${OpenCV_LIBS})

find_package( PCL REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories( ${PCL_LIBRARY_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

find_package (Threads REQUIRED)

add_executable(3dfeaturematcher DescriptorsMatcher/descriptorsmatcher.cpp main.cpp)

target_link_libraries(3dfeaturematcher ${OpenCV_LIBS} ${PCL_LIBRARIES} )

install(TARGETS 3dfeaturematcher RUNTIME DESTINATION bin)
add_subdirectory(DescriptorsMatcher)