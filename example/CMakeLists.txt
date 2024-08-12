# find and initiate all the examples under this folder
file(GLOB_RECURSE EXAMPLEFILES "*.cpp")
igl_include(restricted triangle)
foreach (proj_source ${EXAMPLEFILES})
    get_filename_component(proj_name ${proj_source} NAME_WLE)

    message(STATUS "Compiling project ${proj_name}")

    add_executable(${proj_name} ${proj_source})

    # other dependencies
    target_link_libraries(${proj_name} ${PROJECT_NAME}
            CLI11::CLI11
            igl::core
            igl_restricted::triangle
            polyscope)
endforeach ()