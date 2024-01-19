#include "iostream"
#include "net.hpp"
#include "tic_toc.h"
#include "anms.h"
#include "brief.h"
#include "opencv2/features2d/features2d.hpp"
using namespace std;
using namespace cv;


// 假设每个像素点的64个通道的数据都是按照顺序存储的
 float * GetChannelData( float* p, int width, int height, int x, int y) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		return p + (y * width + x) * 64;
	} else {
		return nullptr; // 当输入的坐标超出范围时返回空指针
	}
}
bool compareKeyPoints(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2) {
	return kp1.response > kp2.response;
}

std::vector<cv::KeyPoint> sortFeaturePoints(std::vector<cv::KeyPoint>& featurePoints) {
	std::sort(featurePoints.begin(), featurePoints.end(), compareKeyPoints);
	return featurePoints;
}


int main()
{
	vector<cv::KeyPoint> last_keypoint;
	Mat last_desc;
	Mat last_image;
	cv::Size size(640, 480);
	std::cout << size.width  <<size.height;
	const char* model_name = "../model/ALIKE.mnn";
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
	cv::Mat hot_pic(cv::Size(640, 480), CV_32FC1);
	// cv::Mat des(cv::Size(640, 480), CV_8UC1, cv::Scalar(0));
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
		std::cout <<descriptors->size()<<std::endl;
		hot_map->printShape();
		descriptors->printShape();
		
		const auto* hotmap_index = (const float*) hot_map->buffer().host;
		int imag_w = hot_pic.size[1];
		hot_pic.forEach<float>([&](float & pixel, const int* position) {
			pixel = (hotmap_index[position[0] * imag_w + position[1]]);
			// std::cout <<" " << pixel;
		});
		// cv::GaussianBlur(hot_pic, hot_pic, cv::Size(3, 3), 5);
		std::vector<cv::KeyPoint> key_points;
		key_points.reserve(640*480);
		// 遍历每个位置，并根据hotmap_index计算关键点位置
		for (int i = 0; i < 480; i++) {
			for (int j = 0; j < 640; j++) {
				float hotmap_value = hotmap_index[i * 640 + j] ;
				if (hotmap_value > 10./255) {
					key_points.emplace_back(j,i,7,-1,hotmap_value); // 将Point对象添加到key_points向量中
				}
			}
		}
		std::vector<cv::KeyPoint> sorted_feature_points = sortFeaturePoints(key_points);
		std::cout <<"sortedFeaturePoints.size() " <<key_points.size() <<std::endl;
		auto* descriptors_index =(float*) descriptors->buffer().host;
		TicToc b;
		vector<cv::KeyPoint> ssc_kp =
				topN(sorted_feature_points, 1000);
		
		Mat desc;
		for (const auto & kp : ssc_kp) {
				float* channel_data = GetChannelData(descriptors_index, 640, 480, kp.pt.x, kp.pt.y);
			cv::Mat des(1, 64, CV_32F, channel_data);
			cv::normalize(des,des, 1, cv::NORM_L2);
			// 将描述子的每个通道数据存放在desc的每一行中
			desc.push_back(des);
		}
		// std::cout <<desc <<std::endl;
		std::cout <<"kepoints.size() " <<ssc_kp.size() <<std::endl;
		std::cout <<"desc    .size() "<<desc.size <<std::endl;
		cv::Mat out;
		cvtColor(image, out, cv::COLOR_GRAY2BGR);
		for (const cv::KeyPoint& kp : ssc_kp) {
			cv::circle(out, kp.pt, 1, cv::Scalar(0, 255, 0), -1); // 在图像上以绿色绘制半径为5的圆
		}
		if(!last_keypoint.empty() && !last_desc.empty())
		{
			// std::cout <<"last  not  empty "<<std::endl;
			cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_L2);
			std::vector<std::vector<cv::DMatch>> matches;
			matcher.knnMatch(last_desc, desc, matches, 2);
			std::vector<cv::DMatch> good_matchs;

			for(auto & matche : matches){
				if(matche[0].distance < 0.7 * matche[1].distance){
					good_matchs.push_back(matche[0]);
				}
			}
			
			//
			// float matchThreshold = 0.1; // 设置匹配阈值
			// std::vector<cv::DMatch> matches;
			// matcher.match(last_desc, desc, matches);
			// std::vector<cv::DMatch> good_matchs;
			// for (const cv::DMatch& match : matches) {
			// 	if (match.distance < matchThreshold) {
			// 		good_matchs.push_back(match);
			// 	}
			// }
			
			
			
			std::cout <<" last_keypoint.size() :"<<last_keypoint.size() <<std::endl;
			std::cout <<" last_desc    .size() :"<<last_desc.size <<std::endl;
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
		// cv::imshow("des",des);
		cv::imshow("out",out);
		
		cv::waitKey(10);
	}
}