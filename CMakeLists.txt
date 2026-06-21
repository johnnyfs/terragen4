cmake_minimum_required(VERSION 4.3)
project(terragen)

file(GLOB_RECURSE sources src/main/*.c src/main/*.h)
file(GLOB_RECURSE sources_test src/test/*.c)
file(GLOB_RECURSE sources_shaders res/*.vert res/*.frag res/*.comp)

find_package(Vulkan REQUIRED COMPONENTS glslc)

set(COMPILED_SHADERS)

foreach(SHADER IN LISTS sources_shaders)
    file(RELATIVE_PATH SHADER_REL "${CMAKE_CURRENT_SOURCE_DIR}/res" "${SHADER}")
    get_filename_component(SHADER_DIR "${SHADER_REL}" DIRECTORY)
    get_filename_component(SHADER_NAME "${SHADER_REL}" NAME)

    set(SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/res/${SHADER_DIR}/compiled")
    set(SPV_FILE "${SPV_DIR}/${SHADER_NAME}.spv")

    add_custom_command(
        OUTPUT "${SPV_FILE}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SPV_DIR}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" "${SHADER}" -o "${SPV_FILE}"
        MAIN_DEPENDENCY "${SHADER}"
        COMMENT "Compiling ${SHADER_REL} to SPIR-V"
        VERBATIM
    )

    list(APPEND COMPILED_SHADERS "${SPV_FILE}")
endforeach()

add_custom_target(CompileShaders ALL DEPENDS ${COMPILED_SHADERS})

find_package(SDL3 REQUIRED)

add_executable(terragen ${sources})
add_dependencies(terragen CompileShaders)

target_compile_options(terragen PUBLIC -std=c23 -Wall -Wfloat-conversion)
target_include_directories(terragen PUBLIC src/main third_party ${SDL3_INCLUDE_DIRS})
target_link_libraries(terragen ${SDL3_LIBRARIES} m)

add_custom_command(
    TARGET terragen POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/res"
            "$<TARGET_FILE_DIR:terragen>/res"
)

enable_testing()

add_executable(terrain_cpu_tests
    src/test/terrain_cpu_tests.c
    src/main/sparse_grid.c
    src/main/chunk_coord.c
    src/main/terrain_config.c
    src/main/terrain_cpu_reference.c
)
target_compile_options(terrain_cpu_tests PUBLIC -std=c23 -Wall -Wfloat-conversion)
target_include_directories(terrain_cpu_tests PUBLIC src/main)
target_link_libraries(terrain_cpu_tests m)
add_test(NAME terrain_cpu_tests COMMAND terrain_cpu_tests)

add_executable(chunk_coord_tests
    src/test/chunk_coord_tests.c
    src/main/chunk_coord.c
    src/main/terrain_config.c
)
target_compile_options(chunk_coord_tests PUBLIC -std=c23 -Wall -Wfloat-conversion)
target_include_directories(chunk_coord_tests PUBLIC src/main)
target_link_libraries(chunk_coord_tests m)
add_test(NAME chunk_coord_tests COMMAND chunk_coord_tests)

add_executable(chunk_cache_tests
    src/test/chunk_cache_tests.c
    src/main/chunk_cache.c
    src/main/chunk_coord.c
    src/main/terrain_config.c
)
target_compile_options(chunk_cache_tests PUBLIC -std=c23 -Wall -Wfloat-conversion)
target_include_directories(chunk_cache_tests PUBLIC src/main)
target_link_libraries(chunk_cache_tests m)
add_test(NAME chunk_cache_tests COMMAND chunk_cache_tests)
