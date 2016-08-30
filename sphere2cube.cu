// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010

#include <vector>
#include <iostream>
#include <stdio.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <opencv2/opencv.hpp>

//////////////////////////////////////////////////
// Utils
//////////////////////////////////////////////////

#define PI2 1.5707963f
#define PI  3.1415927f
#define TAU 6.2831853f

#define GRID_NUM 32

__device__
float myLerp(float a, float b, float k)
{
    //return unsigned char(a * (1.f - k) + b * k);
    return a * (1.f - k) + b * k;
}

__device__
float myClamp(float value, float min, float max)
{
    if      (value < min) return min;
    else if (value > max) return max;
    else                  return value;
}

__device__
void normalize(float *v)
{
    const float k = 1.f / sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] *= k;
    v[1] *= k;
    v[2] *= k;
}


//////////////////////////////////////////////////
// _to_img
//////////////////////////////////////////////////

__device__
void rect_to_img(int h, int w, const float* v, float& i, float& j)
{
    i = h * (       acosf (v[1])        / PI);
    j = w * (0.5f + atan2f(v[0], -v[2]) / TAU);
}

//////////////////////////////////////////////////
// _to_env
//////////////////////////////////////////////////

__device__
void cube_to_env(unsigned f, float i, float j, int h, int w, float* v)
{
    const int p[6][3][3] = {
        {{  0,  0, -1 }, {  0, -1,  0 }, {  1,  0,  0 }},
        {{  0,  0,  1 }, {  0, -1,  0 }, { -1,  0,  0 }},
        {{  1,  0,  0 }, {  0,  0,  1 }, {  0,  1,  0 }},
        {{  1,  0,  0 }, {  0,  0, -1 }, {  0, -1,  0 }},
        {{  1,  0,  0 }, {  0, -1,  0 }, {  0,  0,  1 }},
        {{ -1,  0,  0 }, {  0, -1,  0 }, {  0,  0, -1 }},
    };

    const float y = 2.0f * i / h - 1.0f;
    const float x = 2.0f * j / w - 1.0f;

    v[0] = p[f][0][0] * x + p[f][1][0] * y + p[f][2][0];
    v[1] = p[f][0][1] * x + p[f][1][1] * y + p[f][2][1];
    v[2] = p[f][0][2] * x + p[f][1][2] * y + p[f][2][2];

    normalize(v);
}

//////////////////////////////////////////////////
// Filters
//////////////////////////////////////////////////

__device__
void filter_linear(const unsigned char* srcImg, int srcHeight, int srcWidth, float i, float j, float* p)
{
    const float ii = myClamp(i - 0.5f, 0.0f, srcHeight - 1.0f);
    const float jj = myClamp(j - 0.5f, 0.0f, srcWidth - 1.0f);

    const long  i0 = lrintf(floorf(ii)), i1 = lrintf(ceilf(ii));
    const long  j0 = lrintf(floorf(jj)), j1 = lrintf(ceilf(jj));

    const float di = ii - i0;
    const float dj = jj - j0;

    for(int k=0; k<3; k++) {
        p[k] += myLerp(
            myLerp(srcImg[i0*srcWidth*3 + j0*3 + k], srcImg[i0*srcWidth*3 + j1*3 + k], dj),
            myLerp(srcImg[i1*srcWidth*3 + j0*3 + k], srcImg[i1*srcWidth*3 + j1*3 + k], dj),
            di
            );
    }
}

//////////////////////////////////////////////////
// supersample
//////////////////////////////////////////////////

__global__
void cudaSupersample(
    const unsigned char *srcImg,
    unsigned char *dstImg,
    unsigned srcHeight,
    unsigned srcWidth,
    unsigned dstSize,
    int f
    //int f, int i, int j
)
{
    /*
    //dstImg[threadIdx.x * dstSize + threadIdx.y] = threadIdx.x * dstSize + threadIdx.y;
    dstImg[threadIdx.x * dstSize * 3 + threadIdx.y * 3 + 0] = 66;
    dstImg[threadIdx.x * dstSize * 3 + threadIdx.y * 3 + 1] = 66;
    dstImg[threadIdx.x * dstSize * 3 + threadIdx.y * 3 + 2] = 66;
    */
    int i = GRID_NUM*blockIdx.x + threadIdx.x;
    int j = GRID_NUM*blockIdx.y + threadIdx.y;
    //int f = 0;

    // Pattern
    const int rgss_pattern_size = 4;
    const float rgss_pattern[4][2] = {
        { 0.125f, 0.625f },
        { 0.375f, 0.125f },
        { 0.625f, 0.875f },
        { 0.875f, 0.375f }
    };

    float p[3] = { 0.0f, 0.0f, 0.0f };
    float I, J;

    for(int k=0 ; k<rgss_pattern_size ; k++) {
        const float ii = rgss_pattern[k][0] + i;
        const float jj = rgss_pattern[k][1] + j;

        float v[3] = { 0.0f, 0.0f, 0.0f };
        cube_to_env(f, ii, jj, dstSize, dstSize, v);
        rect_to_img(srcHeight, srcWidth, v, I, J);
        filter_linear(srcImg, srcHeight, srcWidth, I, J, p);
    }
    for(int k=0 ; k<3 ; k++) {
        dstImg[i*dstSize*3 + j*3 + k] = p[k] / rgss_pattern_size;
        //dstImg[i*dstSize*3 + j*3 + k] = 128; // TODO: !?
    }
}

void cudaProcess(
    const unsigned char *srcImg,
    unsigned char *dstImgs[6],
    unsigned srcHeight,
    unsigned srcWidth,
    unsigned dstSize
)
{
    static unsigned char *gpuSrcImg = NULL;
    static unsigned char *gpuDstImgs[6] = {NULL, NULL, NULL, NULL, NULL, NULL};

    //clock_t begin_time = clock(); // TODO: time
    if(!gpuSrcImg) {
        cudaMalloc( (void**)&gpuSrcImg, srcHeight * srcWidth * 3 * sizeof(unsigned char)); // 3: # channels
        for(int f=0 ; f<6 ; f++)
            cudaMalloc( (void**)&gpuDstImgs[f], dstSize * dstSize * 3 * sizeof(unsigned char)); // 3: # channels
    }
    //clock_t end_time = clock(); // TODO: time
    //std::cout << "alloc: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << " sec" << std::endl; // TODO: time

    cudaMemcpy( gpuSrcImg, srcImg, srcHeight * srcWidth * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 dimBlock( dstSize/GRID_NUM, dstSize/GRID_NUM);
	dim3 dimGrid( GRID_NUM, GRID_NUM );
    for(int f=0 ; f<6 ; f++) {
        cudaSupersample<<<dimGrid, dimBlock>>>(gpuSrcImg, gpuDstImgs[f], srcHeight, srcWidth, dstSize, f);
    }

    for(int f=0 ; f<6 ; f++) {
        int result = cudaMemcpy( dstImgs[f], gpuDstImgs[f], dstSize * dstSize * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        //if(result == cudaSuccess)
        //    printf("Copy success\n");
    }

	//cudaFree( gpuSrcImg );
    //for(int f=0 ; f<6 ; f++)
    //    cudaFree( gpuDstImgs[f] );
}

void sphere2cube(cv::Mat& src, std::vector<cv::Mat*>& dst, unsigned size)
{
    for(int i=0 ; i<6 ; i++) {
        dst.push_back(new cv::Mat(size, size, CV_8UC3, cv::Scalar(0, 0, 0)));
    }

    unsigned char *dstImgPtrs[6];
    dstImgPtrs[0] = dst[0]->data;
    dstImgPtrs[1] = dst[1]->data;
    dstImgPtrs[2] = dst[2]->data;
    dstImgPtrs[3] = dst[3]->data;
    dstImgPtrs[4] = dst[4]->data;
    dstImgPtrs[5] = dst[5]->data;
    
    cudaProcess(src.data, dstImgPtrs, src.rows, src.cols, size);
}

void test(int size)
{
    cv::Mat img = cv::imread("input/image_001.tiff", CV_LOAD_IMAGE_UNCHANGED);

    cv::Mat dstImgs0(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat dstImgs1(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat dstImgs2(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat dstImgs3(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat dstImgs4(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat dstImgs5(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    unsigned char *dstImgPtrs[6];
    dstImgPtrs[0] = dstImgs0.data;
    dstImgPtrs[1] = dstImgs1.data;
    dstImgPtrs[2] = dstImgs2.data;
    dstImgPtrs[3] = dstImgs3.data;
    dstImgPtrs[4] = dstImgs4.data;
    dstImgPtrs[5] = dstImgs5.data;

    clock_t begin_time = clock();
    cudaProcess(img.data, dstImgPtrs, img.rows, img.cols, size);
    clock_t end_time = clock();
    std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;

    //imwrite("output/cuda_0.jpg", dstImgs0);
    //imwrite("output/cuda_1.jpg", dstImgs1);
    //imwrite("output/cuda_2.jpg", dstImgs2);
    //imwrite("output/cuda_3.jpg", dstImgs3);
    //imwrite("output/cuda_4.jpg", dstImgs4);
    //imwrite("output/cuda_5.jpg", dstImgs5);

    /*
    for(int i=0 ; i<size ; i++) {
        for(int j=0 ; j<size ; j++) {
            printf("(%d, ", dstImgPtrs[0][i*size*3+j*3+0]);
            printf("%d, ", dstImgPtrs[0][i*size*3+j*3+1]);
            printf("%d) ", dstImgPtrs[0][i*size*3+j*3+2]);
        }
        printf("...\n");
        //printf("\n");
    }
    printf(".\n");
    printf(".\n");
    printf(".\n");
    */
}

void test_sphere2cube(unsigned size)
{
    std::cout << "test_sphere2cube(" << size << ")" << std::endl;
    cv::Mat img = cv::imread("input/image_001.tiff", CV_LOAD_IMAGE_UNCHANGED);
    std::vector<cv::Mat*> dst;

    sphere2cube(img, dst, size);

    imwrite("output/cuda_cube_0.jpg", *(dst[0]));
    imwrite("output/cuda_cube_1.jpg", *(dst[1]));
    imwrite("output/cuda_cube_2.jpg", *(dst[2]));
    imwrite("output/cuda_cube_3.jpg", *(dst[3]));
    imwrite("output/cuda_cube_4.jpg", *(dst[4]));
    imwrite("output/cuda_cube_5.jpg", *(dst[5]));
}

/*
int main()
{
    //for(int i=0 ; i<24 ; i++) {
        //test(32*GRID_NUM);
        //test(32);
        //test(256);
        test_sphere2cube(1024);
    //}
    return 0;
}
*/
