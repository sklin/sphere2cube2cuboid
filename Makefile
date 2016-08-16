all:
	#g++ buildMiniCuboid.cpp cube2cuboid.cpp cutIntoFour.cpp mergeCuboidPieces.cpp mUtils.cpp separateCubeMini.cpp envremap.cpp
	g++ Main.cpp MyServer.cpp lz4.c ImageProcessor.cpp \
	buildMiniCuboid.cpp cube2cuboid.cpp cutIntoFour.cpp mergeCuboidPieces.cpp mUtils.cpp separateCubeMini.cpp sphere2cube.cpp \
    -o run -Wall `pkg-config opencv --cflags --libs` -fopenmp -O3
	#g++ sphere2cube.cpp -Wall `pkg-config opencv --cflags --libs` -fopenmp -O3 -o a.out && ./a.out
server:
	g++ Main.cpp MyServer.cpp lz4.c -o run -std=c++11 -lpthread
