#include "mHeader.hpp"

void cube2cuboid(std::vector<cv::Mat*>& cubeImg, std::vector<cv::Mat*>& cuboidImg)
{
    if(cubeImg.size() < 6) return;
    if(cuboidImg.size() > 0) {
        for(unsigned i=0 ; i<cuboidImg.size() ; i++)
            delete cuboidImg[i];
        cuboidImg.clear();
    }

    int pinpoints[8][3] = {
        {1, 51, 385},  {2, 974, 385},
        {1, 51, 642},  {2, 974, 642},
        {1, 899, 385}, {2, 126, 385},
        {1, 899, 642}, {2, 126, 642}
    };

    int pinpoints2[8][3] = {
        {1, 77, 288},  {2, 759, 385},
        {1, 77, 628},  {2, 759, 579},
        {1, 869, 288}, {2, 309, 385},
        {1, 869, 628}, {2, 309, 579}
    };

    int pinpoints3[8][3] = {
        {1, 45, 365}, {2, 980, 365},
        {1, 45, 625}, {2, 980, 625},
        {1, 870, 365}, {2, 155, 365},
        {1, 870, 625}, {2, 155, 625}
    };

    std::vector<cv::Mat*> miniImgs;
    cutIntoFour(cubeImg, miniImgs);

	std::vector<cv::Mat*> frags, outputs_f, outputs_t, outputs_r;
    for(int i=0 ; i<8 ; i++) {
		int t, v, u, w;
		separateCubeMini(miniImgs, frags, i, pinpoints3[i], t, v, u, w);
        buildMiniCuboid(frags, outputs_f, outputs_t, outputs_r, i, pinpoints3[i], t, v, u, w); // TODO:
    }
	
    for(int i=0 ; i<6 ; i++) {
        cuboidImg.push_back(mergeCuboidPieces(outputs_f, outputs_t, outputs_r, i));
    }
	
    // free the space
    for(unsigned i=0 ; i<miniImgs.size() ; i++) {
        delete miniImgs[i];
    }
	for(unsigned i=0 ; i<frags.size() ; i++) {
		delete frags[i];
	}
	for(unsigned i=0 ; i<outputs_f.size() ; i++) {
		delete outputs_f[i];
		delete outputs_t[i];
		delete outputs_r[i];
	}
}

void test_cube2cuboid() {
    std::vector<cv::Mat*> cubeImg, cuboidImg;
	{ cv::Mat *img = new cv::Mat(); (*img) = cv::imread("cube/cube_0.tiff"); cubeImg.push_back(img); }
	{ cv::Mat *img = new cv::Mat(); (*img) = cv::imread("cube/cube_1.tiff"); cubeImg.push_back(img); }
	{ cv::Mat *img = new cv::Mat(); (*img) = cv::imread("cube/cube_2.tiff"); cubeImg.push_back(img); }
	{ cv::Mat *img = new cv::Mat(); (*img) = cv::imread("cube/cube_3.tiff"); cubeImg.push_back(img); }
	{ cv::Mat *img = new cv::Mat(); (*img) = cv::imread("cube/cube_4.tiff"); cubeImg.push_back(img); }
	{ cv::Mat *img = new cv::Mat(); (*img) = cv::imread("cube/cube_5.tiff"); cubeImg.push_back(img); }
    std::cout << "Start" << std::endl;
    cube2cuboid(cubeImg, cuboidImg);

	//cv::imshow("cuboidImg[0]", *cuboidImg[0]);
	//cv::imshow("cuboidImg[1]", *cuboidImg[1]);
	//cv::imshow("cuboidImg[2]", *cuboidImg[2]);
	//cv::imshow("cuboidImg[3]", *cuboidImg[3]);
	//cv::imshow("cuboidImg[4]", *cuboidImg[4]);
	//cv::imshow("cuboidImg[5]", *cuboidImg[5]);
	//cv::waitKey(0);
    std::cout << "Finish" << std::endl;
}
