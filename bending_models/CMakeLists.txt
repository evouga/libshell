cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

igl_include(restricted triangle)


file(GLOB SRCFILES quickhull/*.cpp quickhull/*.hpp make_geometric_shapes/*.cpp make_geometric_shapes/*.h ShellEnergy/*.h QuadraticExpansionBending.h QuadraticExpansionBending.cpp )	  

add_executable(bending_models bending_model_demo.cpp ${SRCFILES})
# other dependencies
target_link_libraries(bending_models ${PROJECT_NAME}
            optimization
            igl::core
            igl_restricted::triangle
            polyscope
            CLI11::CLI11)

add_executable(trash_can trash_can_demo.cpp ${SRCFILES})
# other dependencies
target_link_libraries(trash_can ${PROJECT_NAME}
            optimization
            igl::core
            igl_restricted::triangle
            polyscope
            CLI11::CLI11)

add_executable(twisted_cylinder twisted_cylinder_demo.cpp ${SRCFILES})
# other dependencies
target_link_libraries(twisted_cylinder ${PROJECT_NAME}
            optimization
            igl::core
            igl_restricted::triangle
            polyscope
            CLI11::CLI11)

add_executable(pin_cloth pin_cloth.cpp ${SRCFILES})
# other dependencies
target_link_libraries(pin_cloth ${PROJECT_NAME}
            optimization
            igl::core
            igl_restricted::triangle    
            polyscope
            CLI11::CLI11)

# Use C++14
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 14)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
