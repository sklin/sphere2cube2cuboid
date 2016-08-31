#include "ImageProcessor.h"
#include "mHeader.hpp"
#include "lz4.h"
//#include <omp.h>
#include <cstdio>

#define BUFFER_SIZE 10240000

char* ImageProcessor::process(int &returnSize) {
    cv::Mat img = cv::imread("input/image_001.tiff", CV_LOAD_IMAGE_UNCHANGED);
    std::vector<cv::Mat*> cubeImg, cuboidImg;

    //double executeStartTime = omp_get_wtime();
    sphere2cube(img, cubeImg, 1024);
    //double executeMiddleTime = omp_get_wtime();
    cube2cuboid(cubeImg, cuboidImg);
    //double executeStopTime = omp_get_wtime();

    // Make packet
    char *buffer = new char [BUFFER_SIZE]; // it will be free in MessageQueue::Pop()

    // Body (Compress)
    //double compressStartTime = omp_get_wtime();
    int compressSize[6] = {0};
    int ptr = 0 + 66; // 66: header's length

    int order[6] = { 4, 5, 0, 1, 2, 3};

    for(int i=0 ; i<6 ; i++) {
        // 3-channel to 4-channel
        cv::Mat mat(cuboidImg[order[i]]->size().height, cuboidImg[order[i]]->size().width, CV_8UC4);
        cv::cvtColor(*cuboidImg[order[i]], mat, CV_BGR2BGRA, 4); 

        // lz4 compress
        int ret = LZ4_compress_default(
            reinterpret_cast<const char*>(mat.data),
            buffer+ptr,
            mat.total() * mat.elemSize(),
            BUFFER_SIZE-ptr);
        ptr += ret;
        compressSize[order[i]] = ret;
        //std::cout << "Before compress, size = " << cuboidImg[i]->total() * cuboidImg[i]->elemSize() << std::endl;
        //std::cout << "After compressed, size = " << ret << std::endl;
        //std::cout << "i: " << order[i] << ", before compress, size = " << cuboidImg[order[i]]->total() * cuboidImg[order[i]]->elemSize() << std::endl;
        //std::cout << "i: " << order[i] << ", compressed, size = " << ret << std::endl;
        //std::cout << "h: " << cuboidImg[order[i]]->size().height << ", w: " << cuboidImg[order[i]]->size().width << std::endl;

        {
            // int LZ4_decompress_safe (const char* source, char* dest, int compressedSize, int maxDecompressedSize);
            char *lz4_buffer = new char[5000000];

            int result = LZ4_decompress_safe(buffer+ptr-ret, lz4_buffer, ret, 5000000);
            std::cout << "result : " << result << std::endl;

            delete[] lz4_buffer;
        }
    }
    //double compressStopTime = omp_get_wtime();

    // Header
    snprintf(buffer, 66, "%010d;%010d;%010d;%010d;%010d;%010d",
        compressSize[order[0]], compressSize[order[1]], compressSize[order[2]],
        compressSize[order[3]], compressSize[order[4]], compressSize[order[5]]
    );
    buffer[65] = ';';
    
    // clean cubeImg and cuboidImg
    for(int i=0 ; i<6 ; i++) {
        delete cubeImg[i];
        delete cuboidImg[i];
    }

    //std::cout << "Spent: " << (compressStopTime - compressStartTime) + (executeStopTime - executeStartTime) << " sec" << std::endl;
    //std::cout << "\tSphere -> Cube: " << executeMiddleTime - executeStartTime << " sec" << std::endl;
    //std::cout << "\tCube -> Cuboid: " << executeStopTime - executeMiddleTime << " sec" << std::endl;
    //std::cout << "\tCompress: " << compressStopTime - compressStartTime << " sec" << std::endl;
    //std::cout << "\tTotal size: " << ptr << std::endl;
    //std::cout << "Finish" << std::endl;

    returnSize = ptr; // TODO:
    return buffer;
    // TODO: remember to clean buffer

    //char *buf = new char[1024];
    //memset(buf, 'A', 1024);
    //return buf;

}
