#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <iostream>

#include <time.h>
#include <cstdio>

#include <omp.h>

#include "mHeader.hpp"

using namespace std;
using namespace cv;

typedef bool (*to_img)(int h, int w, const float* v, unsigned& f, float& i, float& j);
typedef bool (*to_env)(unsigned f, float i, float j, int h, int w, float* v);
typedef void (*filter)(const Mat& img, float i, float j, float* p);

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

//////////////////////////////////////////////////
// Filters
//////////////////////////////////////////////////
void filter_none(const Mat& img, float i, float j, float* p)
{
    return;
}

void filter_linear(const Mat& img, float i, float j, float* p)
{
    const float ii = clamp(i - 0.5f, 0.0f, img.rows - 1.0f);
    const float jj = clamp(j - 0.5f, 0.0f, img.cols - 1.0f);

    const long  i0 = lrintf(floorf(ii)), i1 = lrintf(ceilf(ii));
    const long  j0 = lrintf(floorf(jj)), j1 = lrintf(ceilf(jj));

    const float di = ii - i0;
    const float dj = jj - j0;

    for(int k=0; k<3; k++) {
        p[k] += lerp(
            lerp(img.at<Vec3b>(i0, j0)[k], img.at<Vec3b>(i0, j1)[k], dj),
            lerp(img.at<Vec3b>(i1, j0)[k], img.at<Vec3b>(i1, j1)[k], dj),
            di
            );
    }
}

//////////////////////////////////////////////////
// _to_img
//////////////////////////////////////////////////
bool rect_to_img(int h, int w, const float* v, unsigned& f, float& i, float& j)
{
    f = 0;
    i = h * (       acosf (v[1])        / PI);
    j = w * (0.5f + atan2f(v[0], -v[2]) / TAU);
    return true;
}

bool cube_to_img(int h, int w, const float* v, unsigned& f, float& i, float& j)
{
    //const uchar X = fabsf(v[0]);
    //const uchar Y = fabsf(v[1]);
    //const uchar Z = fabsf(v[2]);
    const float X = fabsf(v[0]);
    const float Y = fabsf(v[1]);
    const float Z = fabsf(v[2]);

    float x;
    float y;

    if      (v[0] > 0 && X >= Y && X >= Z) { f = 0; x = -v[2] / X; y = -v[1] / X; }
    else if (v[0] < 0 && X >= Y && X >= Z) { f = 1; x =  v[2] / X; y = -v[1] / X; }
    else if (v[1] > 0 && Y >= X && Y >= Z) { f = 2; x =  v[0] / Y; y =  v[2] / Y; }
    else if (v[1] < 0 && Y >= X && Y >= Z) { f = 3; x =  v[0] / Y; y = -v[2] / Y; }
    else if (v[2] > 0 && Z >= X && Z >= Y) { f = 4; x =  v[0] / Z; y = -v[1] / Z; }
    else if (v[2] < 0 && Z >= X && Z >= Y) { f = 5; x = -v[0] / Z; y = -v[1] / Z; }
    else return false;

    i = 1.0f + (h - 2) * (y + 1.0f) / 2.0f;
    j = 1.0f + (w - 2) * (x + 1.0f) / 2.0f;

    return true;
}

//////////////////////////////////////////////////
// _to_env
//////////////////////////////////////////////////
bool rect_to_env(unsigned f, float i, float j, int h, int w, float* v)
{
    float lat = PI2 - PI * i / h;
    float lon = TAU * j / w - PI;

    v[0] =  sin(lon) * cos(lat);
    v[1] =             sin(lat);
    v[2] = -cos(lon) * cos(lat);
    return true;
}

bool cube_to_env(unsigned f, float i, float j, int h, int w, float* v)
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
    return true;
}

//////////////////////////////////////////////////
// supersample
//////////////////////////////////////////////////
void supersample(
    vector<Mat*>& src,
    vector<Mat*>& dst,
    const pattern* pat,
    const float* rot,
    filter fil,
    to_img _to_img,
    to_env _to_env,
    unsigned f, int i, int j
    )
{
    //Vec3b& p = dst[f]->at<Vec3b>(i, j);
    float p[3] = { 0.0f, 0.0f, 0.0f };
    unsigned F;
    float I, J;
    int c = 0;

    for(int k=0 ; k<pat->n ; k++) {
        const float ii = pat->p[k].i + i;
        const float jj = pat->p[k].j + j;

        float v[3] = { 0.0f, 0.0f, 0.0f };
        if(_to_env(f, ii, jj, dst[0]->rows, dst[0]->cols, v) &&
            _to_img(src[0]->rows, src[0]->cols, v, F, I, J)) {
            fil(*src[F], I, J, p);

            c++;
        }
    }
    for(int k=0 ; k<3 ; k++) {
        //p[k] /= c;
        dst[f]->at<Vec3b>(i, j)[k] = p[k] / c;
    }
}

//////////////////////////////////////////////////
// process
//////////////////////////////////////////////////
bool process(
    vector<Mat*>& src,
    vector<Mat*>& dst,
    const pattern* pat,
    const float* rot,
    filter fil,
    to_img _to_img,
    to_env _to_env
    )
{
    if( src.size()<1 || dst.size()<1 ) return false;

    int i = 0, j = 0;
    unsigned f = 0;
    #pragma omp parallel for private(j, f)
    for(i=0 ; i<dst[0]->rows ; i++) {
        for(j=0 ; j<dst[0]->cols ; j++) {
            for(f=0 ; f<dst.size() ; f++) {
              supersample(src, dst, pat, rot, fil, _to_img, _to_env, f, i, j);
            }
        }
    }
    return true;
}

//////////////////////////////////////////////////
// sphere2cube
//////////////////////////////////////////////////
void sphere2cube(vector<Mat*>& src, vector<Mat*>& dst, unsigned size)
{
    for(int i=0 ; i<6 ; i++) {
        dst.push_back(new Mat(size, size, CV_8UC3, Scalar(0, 0, 255)));
    }

    const float rot[3] = { 0.0f, 0.0f, 0.0f };
    process(src, dst, &rgss_pattern, rot, filter_linear, rect_to_img, cube_to_env);

    //imwrite("output/cpp_s_0.tiff", *(dst[0]));
    //imwrite("output/cpp_s_1.tiff", *(dst[1]));
    //imwrite("output/cpp_s_2.tiff", *(dst[2]));
    //imwrite("output/cpp_s_3.tiff", *(dst[3]));
    //imwrite("output/cpp_s_4.tiff", *(dst[4]));
    //imwrite("output/cpp_s_5.tiff", *(dst[5]));
}

//////////////////////////////////////////////////
// cube2sphere
//////////////////////////////////////////////////
void cube2sphere(vector<Mat*>& src, vector<Mat*>& dst, unsigned size)
{
    dst.push_back(new Mat(size, size*2, CV_8UC3, Scalar(0, 0, 255)));

    const float rot[3] = { 0.0f, 0.0f, 0.0f };
    process(src, dst, &rgss_pattern, rot, filter_linear, cube_to_img, rect_to_env);
    //process(src, dst, &rgss_pattern, rot, filter_none, cube_to_img, rect_to_env);

    //imwrite("output/cpp_c.tiff", *(dst[0]));
}

//////////////////////////////////////////////////
// test
//////////////////////////////////////////////////
void test_sphere2cube2cuboid(unsigned size)
{
    cout << "test_sphere2cube2cuboid(" << size << ")" << endl;
    vector<Mat*> sphereImg, cubeImg, cuboidImg;

    double prepareStartTime = omp_get_wtime();
    Mat img = imread("input/image_001.tiff", CV_LOAD_IMAGE_UNCHANGED);
    sphereImg.push_back(&img);
    double prepareStopTime = omp_get_wtime();
    printf("Prepare: %.4f sec\n", prepareStopTime - prepareStartTime);

    double executeStartTime = omp_get_wtime();
    sphere2cube(sphereImg, cubeImg, size);
    double executeMiddleTime = omp_get_wtime();
    cube2cuboid(cubeImg, cuboidImg);
    double executeStopTime = omp_get_wtime();
    printf("Execute: %.4f sec\n", executeStopTime - executeStartTime);
    printf("\tSphere -> Cube: %.4f sec\n", executeMiddleTime - executeStartTime);
    printf("\tCube -> Cuboid: %.4f sec\n", executeStopTime - executeMiddleTime);
    cout << "Finish" << endl;

    imwrite("output/cuboid_0.tiff", *(cuboidImg[0]));
    imwrite("output/cuboid_1.tiff", *(cuboidImg[1]));
    imwrite("output/cuboid_2.tiff", *(cuboidImg[2]));
    imwrite("output/cuboid_3.tiff", *(cuboidImg[3]));
    imwrite("output/cuboid_4.tiff", *(cuboidImg[4]));
    imwrite("output/cuboid_5.tiff", *(cuboidImg[5]));
}

void test_cube2sphere(unsigned size)
{
    cout << "test_cube2sphere(" << size << ")" << endl;
    vector<Mat*> src, dst;
    Mat img0 = imread("input/input_0.tiff", CV_LOAD_IMAGE_UNCHANGED);
    Mat img1 = imread("input/input_1.tiff", CV_LOAD_IMAGE_UNCHANGED);
    Mat img2 = imread("input/input_2.tiff", CV_LOAD_IMAGE_UNCHANGED);
    Mat img3 = imread("input/input_3.tiff", CV_LOAD_IMAGE_UNCHANGED);
    Mat img4 = imread("input/input_4.tiff", CV_LOAD_IMAGE_UNCHANGED);
    Mat img5 = imread("input/input_5.tiff", CV_LOAD_IMAGE_UNCHANGED);
    src.push_back(&img0);
    src.push_back(&img1);
    src.push_back(&img2);
    src.push_back(&img3);
    src.push_back(&img4);
    src.push_back(&img5);

    //clock_t tStart = clock();
    double startTime = omp_get_wtime();
    cube2sphere(src, dst, size);
    double stopTime = omp_get_wtime();
    //cout << stopTime - startTime << endl;
    printf("%.4f sec\n", stopTime - startTime);
    //printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

void test_sphere2cube(unsigned size)
{
    cout << "test_sphere2cube(" << size << ")" << endl;
    double prepareStartTime = omp_get_wtime();
    Mat img = imread("input/image_001.tiff", CV_LOAD_IMAGE_UNCHANGED);
    vector<Mat*> src, dst;
    src.push_back(&img);
    double prepareStopTime = omp_get_wtime();
    printf("Prepare: %.4f sec\n", prepareStopTime - prepareStartTime);

    //clock_t tStart = clock();
    double startTime = omp_get_wtime();
    sphere2cube(src, dst, size);
    double stopTime = omp_get_wtime();
    printf("Execute: %.4f sec\n", stopTime - startTime);
    //cout << stopTime - startTime << endl;
    //printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}

int main() {
    test_sphere2cube2cuboid(1024);
    return 0;
}
