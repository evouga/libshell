
add_library(optimization)
message(STATUS "creating target 'optimization'")

file(GLOB_RECURSE INC_FILES "include/*.h")
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
target_sources(optimization PRIVATE ${INC_FILES} ${SRC_FILES})

target_link_libraries(optimization
    PUBLIC
    Eigen3::Eigen
)