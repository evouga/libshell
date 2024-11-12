cmake_minimum_required(VERSION 3.1)
project(bending_models)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

igl_include(restricted triangle)


file(GLOB SRCFILES quickhull/*.cpp quickhull/*.hpp make_geometric_shapes/*.cpp make_geometric_shapes/*.h ShellEnergy/*.h QuadraticExpansionBending.h QuadraticExpansionBending.cpp)	  

add_executable(${PROJECT_NAME} main.cpp ${SRCFILES})
# other dependencies
    target_link_libraries(${proj_name} ${PROJECT_NAME}
            CLI11::CLI11
            igl::core
            igl_restricted::triangle
            polyscope)

# Use C++14
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 14)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
