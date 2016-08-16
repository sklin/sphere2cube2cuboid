#include "ImageProcessor.h"
#include "mHeader.hpp"
#include "lz4.h"
#include <omp.h>
#include <cstdio>

#define BUFFER_SIZE 10240000

char* ImageProcessor::process(int &returnSize) {
    cv::Mat img = cv::imread("input/image_001.tiff", CV_LOAD_IMAGE_UNCHANGED);
    std::vector<cv::Mat*> cubeImg, cuboidImg;

    double executeStartTime = omp_get_wtime();
    sphere2cube(img, cubeImg, 1024);
    double executeMiddleTime = omp_get_wtime();
    cube2cuboid(cubeImg, cuboidImg);
    double executeStopTime = omp_get_wtime();

    // Make packet
    char *buffer = new char [BUFFER_SIZE]; // it will be free in MessageQueue::Pop()

    // Body (Compress)
    double compressStartTime = omp_get_wtime();
    int compressSize[6] = {0};
    int ptr = 0 + 66; // 66: header's length
    for(int i=0 ; i<6 ; i++) {
        int ret = LZ4_compress_default(
            reinterpret_cast<const char*>(cuboidImg[i]->data),
            buffer+ptr,
            cuboidImg[i]->total() * cuboidImg[i]->elemSize(),
            BUFFER_SIZE-ptr);
        ptr += ret;
        compressSize[i] = ret;
        //std::cout << "Before compress, size = " << cuboidImg[i]->total() * cuboidImg[i]->elemSize() << std::endl;
        //std::cout << "After compressed, size = " << ret << std::endl;
    }
    double compressStopTime = omp_get_wtime();

    // Header
    snprintf(buffer, 66, "%010d;%010d;%010d;%010d;%010d;%010d",
        compressSize[0], compressSize[1], compressSize[2],
        compressSize[3], compressSize[4], compressSize[5]
    );
    buffer[65] = ';';
    
    // clean cubeImg and cuboidImg
    for(int i=0 ; i<6 ; i++) {
        delete cubeImg[i];
        delete cuboidImg[i];
    }

    std::cout << "Spent: " << (compressStopTime - compressStartTime) + (executeStopTime - executeStartTime) << " sec" << std::endl;
    std::cout << "\tSphere -> Cube: " << executeMiddleTime - executeStartTime << " sec" << std::endl;
    std::cout << "\tCube -> Cuboid: " << executeStopTime - executeMiddleTime << " sec" << std::endl;
    std::cout << "\tCompress: " << compressStopTime - compressStartTime << " sec" << std::endl;
    std::cout << "\tTotal size: " << ptr << std::endl;
    std::cout << "Finish" << std::endl;

    returnSize = ptr; // TODO:
    return buffer;
    // TODO: remember to clean buffer

    //char *buf = new char[1024];
    //memset(buf, 'A', 1024);
    //return buf;

}
