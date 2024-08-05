include(FetchContent)
FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG "9a6c6f6b50f71d8c4dfc1211947c8f16637b6df5"
)
FetchContent_MakeAvailable(CLI11)