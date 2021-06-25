cmake_minimum_required(VERSION 3.1)
project(libshell)

set (CMAKE_CXX_STANDARD 11)
set(MSVC_RUNTIME "dynamic")
set(LIBIGL_EIGEN_VERSION 3.3.7 CACHE STRING "Eigen version")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Detects whether this is a top-level project
get_directory_property(LIBSHELL_PARENT_DIR PARENT_DIRECTORY)
if(NOT LIBSHELL_PARENT_DIR)
    set(LIBSHELL_TOPLEVEL_PROJECT ON)
else()
    set(LIBSHELL_TOPLEVEL_PROJECT OFF)
endif()

# Build tests
option(LIBSHELL_BUILD_EXEC    "Build libshell executable"   ${LIBSHELL_TOPLEVEL_PROJECT})
option(LIBSHELL_BUILD_TESTS   "Build libshell tests"        ${LIBSHELL_TOPLEVEL_PROJECT})

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add your project files
if(MSVC)
	add_definitions(-DEIGEN_STRONG_INLINE=inline)
endif()

file(GLOB LIBFILES src/*.cpp src/SecondFundamentalForm/*.cpp src/MaterialModel/*.cpp)
add_library(${PROJECT_NAME} STATIC ${LIBFILES})

# install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_LIST_DIR}/lib)
target_include_directories(libshell PRIVATE "${EIGEN3_INCLUDE_DIR}")
target_include_directories(libshell PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")

if(LIBSHELL_BUILD_EXEC)
	find_package(LIBIGL QUIET)

	# libigl
	option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
	option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
	option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          ON)
	option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
	
	file(GLOB EXAMPLEFILES example/*.cpp)
	add_executable(example_${PROJECT_NAME} ${EXAMPLEFILES})
	target_link_libraries(example_${PROJECT_NAME} ${PROJECT_NAME} igl::core igl::opengl_glfw igl::opengl_glfw_imgui)
endif()


if(LIBSHELL_BUILD_TESTS)
	file(GLOB TESTFILES tests/*.cpp)
	add_executable(tests_${PROJECT_NAME} ${TESTFILES})
	target_link_libraries(tests_${PROJECT_NAME} ${PROJECT_NAME})
endif()
