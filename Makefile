all:
	nvcc -ccbin g++ Main.cpp MyServer.cpp lz4.c ImageProcessor.cpp \
	buildMiniCuboid.cpp cube2cuboid.cpp cutIntoFour.cpp mergeCuboidPieces.cpp mUtils.cpp separateCubeMini.cpp sphere2cube.cu \
    -o run `pkg-config opencv --cflags --libs` -O3 \
    -I/home/sklin/NVIDIA_CUDA-8.0_Samples/common/inc
