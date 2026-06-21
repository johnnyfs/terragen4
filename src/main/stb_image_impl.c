/* Single translation unit that compiles the vendored stb_image implementation.
 * Only PNG is needed (terrain material textures ship as PNG). */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"
