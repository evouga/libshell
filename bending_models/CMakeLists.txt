cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

igl_include(restricted triangle)


file(GLOB SRCFILES quickhull/*.cpp quickhull/*.hpp make_geometric_shapes/*.cpp make_geometric_shapes/*.h ShellEnergy/*.h QuadraticExpansionBending.h QuadraticExpansionBending.cpp )	  

add_executable(bending_models main.cpp ${SRCFILES})
# other dependencies
target_link_libraries(bending_models ${PROJECT_NAME}
            optimization
            igl::core
            igl_restricted::triangle
            polyscope
            CLI11::CLI11)

# Use C++14
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 14)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
