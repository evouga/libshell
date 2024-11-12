cmake_minimum_required(VERSION 3.1)
project(libshell)

set (CMAKE_CXX_STANDARD 17)
set(MSVC_RUNTIME "dynamic")
set(LIBIGL_EIGEN_VERSION 3.3.7 CACHE STRING "Eigen version")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(libigl)

# Add your project files
if(MSVC)
	add_definitions(-DEIGEN_STRONG_INLINE=inline)
endif()

file(GLOB LIBFILES src/*.cpp src/SecondFundamentalForm/*.cpp src/MaterialModel/*.cpp)
add_library(${PROJECT_NAME} STATIC ${LIBFILES} ${OPTFILES})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_LIST_DIR}/lib)

add_subdirectory(optimization)
add_subdirectory(example)

file(GLOB TESTFILES tests/*.cpp)
add_executable(tests_${PROJECT_NAME} ${TESTFILES})
target_link_libraries(tests_${PROJECT_NAME} ${PROJECT_NAME} Eigen3::Eigen)
