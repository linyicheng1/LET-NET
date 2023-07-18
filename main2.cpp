#include "iostream"
#include <MNN/ImageProcess.hpp>
#include "opencv2/opencv.hpp"
#include "net.hpp"
#include "tic_toc.h"
using namespace  std;
int main()
{
	const char* superpoint_model_name = "../model/mnn/model640*480.mnn";
	Net net = Net(superpoint_model_name);
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
		if (image.empty())
			cout<<"images input error! check please."<<endl;
		
		cv::resize(image, image, cv::Size(640, 480), 0, 0);
		TicToc a;
		net.Inference(image);  //推理
		auto hotmap = net.GetScoresValue();
		auto descriptors = net.GetDescriptorsValueOnly();
		hotmap->host<float>();
		vector<int> hotmap_shape(hotmap->shape()); //获取维度
		
		const auto* hotmap_index = (const float*) hotmap->buffer().host;
		cv::Mat hot_pic(cv::Size(640,480), CV_8UC1,cv::Scalar(0));
		int imag_w = hot_pic.size[1];
		for(int i = 0 ; i<hot_pic.size[0] ;++i){
			for(int j = 0 ; j<hot_pic.size[1] ; ++j){
				hot_pic.at<uchar>(i,j) = hotmap_index[i*imag_w+j]*255;
			}
		}
		// cv::imshow("2",hotPic);
		const auto* descriptors_index = (const float*) descriptors->buffer().host;
		cv::Mat des(cv::Size(640,480), CV_8UC3,cv::Scalar(0));
		imag_w = des.size[1];
		for(int i = 0 ; i<des.size[0] ;++i){
			for(int j = 0 ; j<des.size[1] ; j+=3){
				des.at<cv::Vec3b>(i,j) = cv::Vec3b(descriptors_index[i*imag_w+j]*255,
				                                   descriptors_index[i*imag_w+j+307200]*255,
				                                   descriptors_index[i*imag_w+j+614400]*255);
			}
		}
		std::cout <<" ======================="<<  a.toc() <<std::endl;
		// cv::imshow("4",Des);
		// cv::imshow("input", image);
		cv::waitKey(1);
	}
}