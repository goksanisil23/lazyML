## Build
- Extract the MNIST dataset to the path of this binary.
```sh
clang++ mnist.cpp -o mnist -I/usr/include/eigen3 -I/usr/include/SDL2/ -lSDL2 -O3 -Wall -std=c++17
```
