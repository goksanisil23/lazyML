#include <SDL2/SDL.h>

// Define the image dimensions
const int IMAGE_WIDTH  = 28;
const int IMAGE_HEIGHT = 28;

int render(unsigned char *image_buffer)
{
    // unsigned char image_buffer[IMAGE_WIDTH * IMAGE_HEIGHT * 4];
    // Initialize SDL
    SDL_Init(SDL_INIT_VIDEO);

    // Create a new window
    SDL_Window *window = SDL_CreateWindow("Image Rendering",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          IMAGE_WIDTH,
                                          IMAGE_HEIGHT,
                                          SDL_WINDOW_SHOWN);

    // Create a renderer for the window
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);

    // Create a texture to hold the image
    SDL_Texture *texture =
        SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, IMAGE_WIDTH, IMAGE_HEIGHT);

    // // Populate the image buffer with sample data
    // populateImageBuffer(image_buffer);

    // Update the texture with the image data
    SDL_UpdateTexture(texture, nullptr, image_buffer, IMAGE_WIDTH * sizeof(unsigned char) * 4);

    // Clear the renderer
    SDL_RenderClear(renderer);

    // Copy the texture onto the renderer
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);

    // Render the result
    SDL_RenderPresent(renderer);

    // Wait for the window to be closed
    bool      quit = false;
    SDL_Event event;
    while (!quit)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
            {
                quit = true;
            }
            else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_SPACE)
            {
                quit = true;
            }
        }
    }

    // Cleanup resources
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
