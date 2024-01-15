# LET-NET: A lightweight CNN network for sparse corners extraction and tracking

LET-NET implements an extremely lightweight network for feature point extraction and image consistency computation. The network can process a 240 x 320 image on a CPU in about 5ms. Combined with LK optical flow, it breaks the assumption of brightness consistency and performs well on dynamic lighting as well as blurred images.


## 1. Prerequisites 

- OpenCV (https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html)
- mnn  (https://mnn-docs.readthedocs.io/en/latest/index.html)
> Notes: After installing mnn, you need to change the path in CMakeLists.txt

```
set(MNN "/home/moi/APP/include/mnn2" CACHE STRING "MNN library path")
```
## 2. Build 

```
mkdir build && cd build
cmake .. && make -j4
```

