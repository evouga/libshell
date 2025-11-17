cmake_minimum_required(VERSION 4.0)
cmake_policy(SET CMP0169 OLD)
project(libshell)

set (CMAKE_CXX_STANDARD 11)
set(MSVC_RUNTIME "dynamic")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(libigl)

# Add your project files
if(MSVC)
	add_definitions(-DEIGEN_STRONG_INLINE=inline)
	add_compile_options(/bigobj)
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
