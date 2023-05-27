## Build
- Extract the MNIST dataset to the path of this binary.
```sh
g++ mnist.cpp -o mnist -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc
```
