#include "iostream"
#include <MNN/ImageProcess.hpp>
#include "opencv2/opencv.hpp"
#include "net.hpp"
#include "tic_toc.h"
using namespace  std;
int main()
{
	const char* model_name = "../model/mnn/model640*480.mnn";
	Net net = Net(model_name);
	cv::VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		std::cerr << "ERROR!!Unable to open camera\n";
		return -1;
	}
	cv::Mat img;
	while (true)
	{
		cap >> img;
		cv::Mat image = img;
		if (image.empty()) {
			cout << "images input error! check please." << endl;
		}
		cv::resize(image, image, cv::Size(640, 480), 0, 0);
		TicToc a;
		net.Inference(image);  //推理
		auto hotmap = net.GetScoresValue();
		auto descriptors = net.GetDescriptorsValueOnly();
		hotmap->host<float>();
		vector<int> hotmap_shape(hotmap->shape()); //获取维度
		const float* hotmap_index = hotmap->host<float>();
		cv::Mat hot_pic(cv::Size(640, 480), CV_8UC1, cv::Scalar(0));
		int imag_w = hot_pic.size[1];
		hot_pic.forEach<uchar>([&](uchar& pixel, const int* position) {
			pixel = hotmap_index[position[0] * imag_w + position[1]] * 255;
		});
		
		const float* descriptors_index = descriptors->host<float>();
		cv::Mat des(cv::Size(640, 480), CV_8UC3, cv::Scalar(0));
		imag_w = des.size[1];
		des.forEach<cv::Vec3b>([&](cv::Vec3b& pixel, const int* position) {
			int pixel_index = position[0] * imag_w + position[1];
			int channel_offset = imag_w * 3;
			pixel[0] = static_cast<uchar>(descriptors_index[pixel_index] * 255);
			pixel[1] = static_cast<uchar>(descriptors_index[pixel_index + channel_offset] * 255);
			pixel[2] = static_cast<uchar>(descriptors_index[pixel_index + channel_offset * 2] * 255);
		});
		std::cout <<" ======================="<<  a.toc() <<std::endl;
		cv::imshow("2",hot_pic);
		cv::imshow("4",des);
		cv::imshow("input", image);
		cv::waitKey(1);
	}
}