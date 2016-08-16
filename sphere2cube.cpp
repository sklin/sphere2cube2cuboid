#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <iostream>

#include <omp.h>
#include "mHeader.hpp"

//////////////////////////////////////////////////
// Utils
//////////////////////////////////////////////////

#define PI2 1.5707963f
#define PI  3.1415927f
#define TAU 6.2831853f

static inline float lerp(float a, float b, float k)
{
    return uchar(a * (1.f - k) + b * k);
}

static inline float clamp(float value, float min, float max)
{
    if      (value < min) return min;
    else if (value > max) return max;
    else                  return value;
}

static inline void normalize(float *v)
{
    const float k = 1.f / sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] *= k;
    v[1] *= k;
    v[2] *= k;
}

//////////////////////////////////////////////////
// A pattern structure represents a supersampling pattern.
//////////////////////////////////////////////////

/*
struct point
{
    float i;
    float j;
};

struct pattern
{
    int    n;
    point *p;
};

static point rgss_points[] = {
    { 0.125f, 0.625f },
    { 0.375f, 0.125f },
    { 0.625f, 0.875f },
    { 0.875f, 0.375f },
};

static const pattern rgss_pattern = {  4, rgss_points };
*/

//////////////////////////////////////////////////
// Filters
//////////////////////////////////////////////////

void filter_linear(const cv::Mat* img, float i, float j, float* p)
{
    const float ii = clamp(i - 0.5f, 0.0f, img->rows - 1.0f);
    const float jj = clamp(j - 0.5f, 0.0f, img->cols - 1.0f);

    const long  i0 = lrintf(floorf(ii)), i1 = lrintf(ceilf(ii));
    const long  j0 = lrintf(floorf(jj)), j1 = lrintf(ceilf(jj));

    const float di = ii - i0;
    const float dj = jj - j0;

    for(int k=0; k<3; k++) {
        p[k] += lerp(
            lerp(img->at<cv::Vec3b>(i0, j0)[k], img->at<cv::Vec3b>(i0, j1)[k], dj),
            lerp(img->at<cv::Vec3b>(i1, j0)[k], img->at<cv::Vec3b>(i1, j1)[k], dj),
            di
            );
    }
}

//////////////////////////////////////////////////
// _to_img
//////////////////////////////////////////////////

void rect_to_img(int h, int w, const float* v, float& i, float& j)
{
    i = h * (       acosf (v[1])        / PI);
    j = w * (0.5f + atan2f(v[0], -v[2]) / TAU);
}

//////////////////////////////////////////////////
// _to_env
//////////////////////////////////////////////////

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
// supersample
//////////////////////////////////////////////////
void supersample(
    cv::Mat* src,
    cv::Mat* dst,
    unsigned f, int i, int j
    )
{
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
        cube_to_env(f, ii, jj, dst->rows, dst->cols, v);
        rect_to_img(src->rows, src->cols, v, I, J);
        filter_linear(src, I, J, p);
    }
    for(int k=0 ; k<3 ; k++) {
        dst->at<cv::Vec3b>(i, j)[k] = p[k] / (rgss_pattern_size);
    }
}

//////////////////////////////////////////////////
// process
//////////////////////////////////////////////////
bool process(
    cv::Mat* src,
    std::vector<cv::Mat*>& dst
    )
{
    if( dst.size()<1 ) return false;

    int i, j;
    unsigned f;
    #pragma omp parallel for private(j, f)
    for(i=0 ; i<dst[0]->rows ; i++) {
        for(j=0 ; j<dst[0]->cols ; j++) {
            for(f=0 ; f<dst.size() ; f++) {
                supersample(src, dst[f], f, i, j);
            }
        }
    }
    return true;
}

//////////////////////////////////////////////////
// sphere2cube
//////////////////////////////////////////////////

void sphere2cube(cv::Mat& src, std::vector<cv::Mat*>& dst, unsigned size)
{
    for(int i=0 ; i<6 ; i++) {
        dst.push_back(new cv::Mat(size, size, CV_8UC3, cv::Scalar(0, 0, 0)));
    }

    process(&src, dst);
}

//////////////////////////////////////////////////
// test
//////////////////////////////////////////////////

void test_sphere2cube(unsigned size)
{
    std::cout << "test_sphere2cube(" << size << ")" << std::endl;
    cv::Mat img = cv::imread("input/image_001.tiff", CV_LOAD_IMAGE_UNCHANGED);
    std::vector<cv::Mat*> dst;

    double startTime = omp_get_wtime();
    sphere2cube(img, dst, size);
    double stopTime = omp_get_wtime();
    std::cout << stopTime - startTime << " sec" << std::endl;

    imwrite("output/dedicate_cube_0.tiff", *(dst[0]));
    imwrite("output/dedicate_cube_1.tiff", *(dst[1]));
    imwrite("output/dedicate_cube_2.tiff", *(dst[2]));
    imwrite("output/dedicate_cube_3.tiff", *(dst[3]));
    imwrite("output/dedicate_cube_4.tiff", *(dst[4]));
    imwrite("output/dedicate_cube_5.tiff", *(dst[5]));
}

void test_sphere2cube2cuboid(unsigned size)
{
    std::cout << "test_sphere2cube2cuboid(" << size << ")" << std::endl;
    cv::Mat img = cv::imread("input/image_001.tiff", CV_LOAD_IMAGE_UNCHANGED);
    std::vector<cv::Mat*> cubeImg, cuboidImg;

    double executeStartTime = omp_get_wtime();
    sphere2cube(img, cubeImg, size);
    double executeMiddleTime = omp_get_wtime();
    cube2cuboid(cubeImg, cuboidImg);
    double executeStopTime = omp_get_wtime();

    std::cout << "Execute: " << executeStopTime - executeStartTime << " sec" << std::endl;
    std::cout << "\tSphere -> Cube: " << executeMiddleTime - executeStartTime << " sec" << std::endl;
    std::cout << "\tCube -> Cuboid: " << executeStopTime - executeMiddleTime << " sec" << std::endl;
    std::cout << "Finish" << std::endl;

    imwrite("output/dedicate_cuboid_0.tiff", *(cuboidImg[0]));
    imwrite("output/dedicate_cuboid_1.tiff", *(cuboidImg[1]));
    imwrite("output/dedicate_cuboid_2.tiff", *(cuboidImg[2]));
    imwrite("output/dedicate_cuboid_3.tiff", *(cuboidImg[3]));
    imwrite("output/dedicate_cuboid_4.tiff", *(cuboidImg[4]));
    imwrite("output/dedicate_cuboid_5.tiff", *(cuboidImg[5]));
}

//////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////

/*
int main()
{
    test_sphere2cube2cuboid(1024);
    return 0;
}
*/
