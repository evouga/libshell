# find and initiate all the examples under this folder
file(GLOB_RECURSE EXAMPLEFILES "*.cpp")
include(polyscope)

foreach (proj_source ${EXAMPLEFILES})
    get_filename_component(exe_name ${proj_source} NAME_WLE)

    message(STATUS "Compiling project ${exe_name}")

    add_executable(${exe_name} ${proj_source} ${OPTFILES})

    # other dependencies
    target_link_libraries(${exe_name} ${PROJECT_NAME}
            optimization
            igl::core
            igl_restricted::triangle
            polyscope
            CLI11::CLI11)
endforeach ()