include(FetchContent)
FetchContent_Declare(
        polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG 2cd38b7496a1e8ff79f615920de2980295bedef7
)

FetchContent_GetProperties(polyscope)
if(NOT polyscope_POPULATED)
    # Fetch the content using previously declared details
    FetchContent_Populate(polyscope)
    message(STATUS "polyscope_SOURCE_DIR: ${polyscope_SOURCE_DIR}")
    message(STATUS "polyscope_BINARY_DIR: ${polyscope_BINARY_DIR}")
    add_subdirectory(${polyscope_SOURCE_DIR} ${polyscope_BINARY_DIR})
endif()
FetchContent_MakeAvailable(polyscope)

