cmake_minimum_required(VERSION 4.3)
project(terragen)

file(GLOB_RECURSE sources src/main/*.c src/main/*.h)
file(GLOB_RECURSE sources_test src/test/*.c)
file(GLOB_RECURSE sources_shaders res/*.vert res/*.frag res/*.comp)
file(GLOB_RECURSE shader_includes res/shaders/*.glsl)

find_package(Vulkan QUIET COMPONENTS glslc)
set(GLSLC_EXECUTABLE "${Vulkan_GLSLC_EXECUTABLE}")
if(NOT GLSLC_EXECUTABLE)
    find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/bin")
endif()
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "Could not find glslc. Install shaderc (for example: brew install shaderc).")
endif()

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
        COMMAND "${GLSLC_EXECUTABLE}" -I "${CMAKE_CURRENT_SOURCE_DIR}/res/shaders" "${SHADER}" -o "${SPV_FILE}"
        MAIN_DEPENDENCY "${SHADER}"
        DEPENDS ${shader_includes}
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
target_include_directories(terragen PUBLIC src/main third_party)
if(TARGET SDL3::SDL3)
    target_link_libraries(terragen SDL3::SDL3)
else()
    target_include_directories(terragen PUBLIC ${SDL3_INCLUDE_DIRS})
    target_link_libraries(terragen ${SDL3_LIBRARIES})
endif()
target_link_libraries(terragen m)

if(APPLE)
    set(TERRAGEN_SHADERCROSS_ROOT "" CACHE PATH "SDL3_shadercross install prefix")
    set(TERRAGEN_SHADERCROSS_PREFIXES)
    if(TERRAGEN_SHADERCROSS_ROOT)
        list(APPEND TERRAGEN_SHADERCROSS_PREFIXES "${TERRAGEN_SHADERCROSS_ROOT}")
    endif()
    list(APPEND TERRAGEN_SHADERCROSS_PREFIXES
        "${CMAKE_CURRENT_SOURCE_DIR}/../SDL_shadercross-terragen/install"
        "$ENV{HOME}/.local"
    )

    find_package(SDL3_shadercross CONFIG QUIET PATHS ${TERRAGEN_SHADERCROSS_PREFIXES})
    if(NOT SDL3_shadercross_FOUND)
        message(FATAL_ERROR "Could not find SDL3_shadercross. Build/install SDL_shadercross and add its prefix to CMAKE_PREFIX_PATH or TERRAGEN_SHADERCROSS_ROOT.")
    endif()
    if(NOT TARGET SDL3_shadercross::SDL3_shadercross)
        message(FATAL_ERROR "SDL3_shadercross was found, but target SDL3_shadercross::SDL3_shadercross is missing.")
    endif()
    target_compile_definitions(terragen PRIVATE TERRAGEN_USE_SHADERCROSS=1)
    target_link_libraries(terragen SDL3_shadercross::SDL3_shadercross)
endif()

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

add_executable(terrain_scene_tests
    src/test/terrain_scene_tests.c
    src/main/chunk_coord.c
    src/main/materials.c
    src/main/terrain_config.c
    src/main/terrain_scene.c
)
target_compile_options(terrain_scene_tests PUBLIC -std=c23 -Wall -Wfloat-conversion)
target_include_directories(terrain_scene_tests PUBLIC src/main)
target_link_libraries(terrain_scene_tests m)
add_test(NAME terrain_scene_tests COMMAND terrain_scene_tests)
set_tests_properties(terrain_scene_tests PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
