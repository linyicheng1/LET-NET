#include "iostream"
#include "net.hpp"
#include "tic_toc.h"
#include "anms.h"
#include "brief.h"
using namespace std;
using namespace cv;
int main()
{
	const int npoints = 512;
	const auto *pattern0 = (const Point *)bit_pattern_31_;
	std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
	std::vector<int> umax;
	umax.resize(HALF_PATCH_SIZE + 1);
	int v,                                                    // 循环辅助变量
	v0,                                                   // 辅助变量
	vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);  // 计算圆的最大行号，+1应该是把中间行也给考虑进去了
	int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
	const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
	for (v = 0; v <= vmax; ++v)
	{
		umax[v] = cvRound(sqrt(hp2 - v * v));
	}
	for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
		while (umax[v0] == umax[v0 + 1]) ++v0;
		umax[v] = v0;
		++v0;
	}
	
	vector<cv::KeyPoint> last_keypoint;
	Mat last_desc;
	Mat last_image;
	cv::Size size(640, 480);
	std::cout << size.width  <<size.height;
	const char* model_name = "../model/letnet_out.mnn";
	Net net = Net(model_name);
	cv::VideoCapture cap;
	cap.open("/home/moi/left_output_video.avi");
	// cap.open(0);
	if (!cap.isOpened())
	{
		std::cerr << "ERROR!!Unable to open camera\n";
		return -1;
	}
	cv::Mat img;
	cv::Mat hot_pic(cv::Size(640, 480), CV_8UC1, cv::Scalar(0));
	cv::Mat des(cv::Size(640, 480), CV_8UC1, cv::Scalar(0));
	while (true)
	{
		cap >> img;
		cv::Mat image;
		if(img.channels() ==3){
			cvtColor(img, image, cv::COLOR_BGR2GRAY);
		}
		else {
			image = img;
		}
		if (image.empty()) {
			cout << "images input error! check please." << endl;
		}
		cv::resize(image, image, cv::Size(640, 480), 0, 0);
		TicToc a;
		net.Inference(image);  //推理
		auto hot_map = net.GetScoresValue();
		auto descriptors = net.GetDescriptorsValue();
		const auto* hotmap_index = (const float*) hot_map->buffer().host;
		int imag_w = hot_pic.size[1];
		hot_pic.forEach<uchar>([&](uchar& pixel, const int* position) {
			pixel = static_cast<uchar>(hotmap_index[position[0] * imag_w + position[1]]  *  255);
		});
		std::vector<cv::KeyPoint> key_points;
		key_points.reserve(640*480);
		// 遍历每个位置，并根据hotmap_index计算关键点位置
		for (int i = 0; i < 480; i++) {
			for (int j = 0; j < 640; j++) {
				float hotmap_value = hotmap_index[i * 640 + j] ;
				if (hotmap_value > 1./255) {
					key_points.emplace_back(j,i,7,-1,hotmap_value*255); // 将Point对象添加到key_points向量中
				}
			}
		}
		const auto* descriptors_index =(const float*) descriptors->buffer().host;
		imag_w = des.size[1];
		des.forEach<uchar>([&](uchar& pixel, const int* position) {
			pixel = static_cast<uchar>(descriptors_index[position[0] * imag_w + position[1]] * 255);
		});
		
		TicToc b;
		vector<cv::KeyPoint> ssc_kp =
				ssc(key_points, 1000, 0.1, 640, 480);
		
		cv::Mat out;
		cvtColor(image, out, cv::COLOR_GRAY2BGR);
		for (const cv::KeyPoint& kp : ssc_kp) {
			cv::circle(out, kp.pt, 1, cv::Scalar(0, 255, 0), -1); // 在图像上以绿色绘制半径为5的圆
		}
		computeOrientation(des,  // 对应的图层的图像
		                   ssc_kp,    // 这个图层中提取并保留下来的特征点容器
		                   umax);                  // 以及PATCH的横坐标边界
		
		Mat desc = cv::Mat(ssc_kp.size(), 32, CV_8U);
		
		GaussianBlur(des,                    // 源图像
		             des,                   // 输出图像
		             Size(7, 7),           // 高斯滤波器kernel大小，必须为正的奇数
		             2,                    // 高斯滤波在x方向的标准差
		             2,                    // 高斯滤波在y方向的标准差
		             BORDER_REFLECT_101);  // 边缘拓展点插值类型
		//
		computeDescriptors(des,  // 高斯模糊之后的图层图像
		                   ssc_kp,   // 当前图层中的特征点集合
		                   desc,        // 存储计算之后的描述子
		                   pattern);    // 随机采样点集
	   // std::cout <<desc <<std::endl;
		if(!last_keypoint.empty() && !last_desc.empty())
		{
			// std::cout <<"last  not  empty "<<std::endl;
			cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING2);
			std::vector<std::vector<cv::DMatch>> matches;
			matcher.knnMatch(last_desc, desc, matches, 2);
			std::vector<cv::DMatch> good_matchs;
			
			for(auto & matche : matches){
				if(matche[0].distance < 0.75 * matche[1].distance){
					good_matchs.push_back(matche[0]);
				}
			}
			std::cout << "good_matchs:" << good_matchs.size() << std::endl;
			Mat matchimages;
			drawMatches(last_image,last_keypoint,image,ssc_kp,good_matchs,matchimages);
			imshow("matchImages",matchimages);
		}
		last_keypoint = ssc_kp;
		last_desc = desc;
		last_image =image;
		// std::cout <<" ======================="<<  a.toc() <<std::endl;
		cv::imshow("hot_pic",hot_pic);
		cv::imshow("des",des);
		cv::imshow("out",out);
		
		cv::waitKey(-1);
	}
}