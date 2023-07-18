#include "iostream"
#include<MNN/Interpreter.hpp>
#include<MNN/ImageProcess.hpp>
#include "opencv2/opencv.hpp"
#include "net.hpp"
#include "tic_toc.h"
using namespace  std;
int main()
{
	const char* superpoint_model_name = "../model/mnn/model640*480.mnn";
	cout<<"load model from "<<superpoint_model_name<<endl;
	Net net = Net(superpoint_model_name);
	shared_ptr<MNN::Tensor> descFinalTensor1;
	vector<pair<int,int>> keyPsVecs1;
	cv::VideoCapture cap;   //声明相机捕获对象
	cap.open(0); //打开相机
	if (!cap.isOpened()) //判断相机是否打开
	{
		std::cerr << "ERROR!!Unable to open camera\n";
		return -1;
	}
	cv::Mat img;
	while (1)
	{
		cap >> img; //以流形式捕获图像
		cv::Mat image = img;
		if (image.empty()){
			cout<<"images input error! check please."<<endl;
		}
		cv::resize(image, image, cv::Size(640, 480), 0, 0);
		TicToc a;
		net.Inference(image);  //推理
		auto hotmap = net.GetScoresValue();
		auto descriptors = net.GetDescriptorsValueOnly();
		hotmap->host<float>();
		vector<int> hotmap_shape(hotmap->shape()); //获取维度
		
		const float* hotmapIndex = (const float*) hotmap->buffer().host;
		// hotmap可视化：其实也显示不出来啥，就几个分布的亮点。
		cv::Mat hotPic(cv::Size(640,480), CV_8UC1,cv::Scalar(0));
		int imag_w = hotPic.size[1];
		for(int i = 0 ; i<hotPic.size[0] ;++i){
			for(int j = 0 ; j<hotPic.size[1] ; ++j){
				hotPic.at<uchar>(i,j) = hotmapIndex[i*imag_w+j]*255;
			}
		}
		// cv::imshow("2",hotPic);
		const float* DescriptorsIndex = (const float*) descriptors->buffer().host;
		cv::Mat Des(cv::Size(640,480), CV_8UC3,cv::Scalar(0));
		// cout<<"shape1 : "<<Des.size[0]<<" "<<Des.size[1]<<endl;
		imag_w = Des.size[1];
		for(int i = 0 ; i<Des.size[0] ;++i){
			for(int j = 0 ; j<Des.size[1] ; j+=3){
				Des.at<cv::Vec3b>(i,j) = cv::Vec3b(DescriptorsIndex[i*imag_w+j]*255,
				                                   DescriptorsIndex[i*imag_w+j+307200]*255,
				                                   DescriptorsIndex[i*imag_w+j+614400]*255);
			}
		}
		std::cout <<" ======================="<<  a.toc() <<std::endl;
		// cv::imshow("4",Des);
		// cv::imshow("input", image);
		cv::waitKey(10);
	}
}