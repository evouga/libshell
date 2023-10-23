cmake_minimum_required(VERSION 3.9)

find_package(Enzyme REQUIRED CONFIG)

# Trade off flexibility for correctness: compilation *will* fail if using a non
# clang compiler, or a different clang/LLVM version than the one used to build
# Enzyme. Compilers can only be configured before a project() call.
message("LLVM Version: " ${Enzyme_LLVM_VERSION_MAJOR})
message("Found LLVM at: " ${Enzyme_LLVM_BINARY_DIR})
set(CMAKE_C_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang")
set(CMAKE_CXX_COMPILER "${Enzyme_LLVM_BINARY_DIR}/bin/clang++")

project(libshell)

get_property(importTargetsAfter DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
message("imported targets ${importTargetsAfter}")

set(CMAKE_CXX_STANDARD 11)
set(LIBIGL_EIGEN_VERSION 3.3.7 CACHE STRING "Eigen version")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(libigl)
igl_include(glfw)
igl_include(imgui)

file(GLOB LIBFILES src/*.cpp src/SecondFundamentalForm/*.cpp src/MaterialModel/*.cpp)
add_library(${PROJECT_NAME} STATIC ${LIBFILES})
target_link_libraries(${PROJECT_NAME} PUBLIC igl::core LLDEnzymeFlags)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_LIST_DIR}/lib)

file(GLOB EXAMPLEFILES example/*.cpp)
add_executable(example_${PROJECT_NAME} ${EXAMPLEFILES})
target_link_libraries(example_${PROJECT_NAME} PUBLIC ${PROJECT_NAME} igl::core igl::glfw igl::imgui LLDEnzymeFlags)

file(GLOB TESTFILES tests/*.cpp)
add_executable(tests_${PROJECT_NAME} ${TESTFILES})
target_link_libraries(tests_${PROJECT_NAME} PUBLIC ${PROJECT_NAME} igl::core LLDEnzymeFlags)
