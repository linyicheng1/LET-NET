#include "iostream"
#include "net.hpp"
#include "tic_toc.h"
#include "anms.h"
#include "opencv2/features2d/features2d.hpp"

using namespace std;
using namespace cv;

#if USE_ORB
#include "orb_extractor.h"
orb_exteactor::ORBextractor *mpIniORBextractor = new orb_exteactor::ORBextractor(1000,1.2,8,20,5);
# endif

Scalar RandomColor(int64 seed) {
	RNG rng(seed);
	int icolor = (unsigned) rng;
	return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

Mat ConnectPointVectorOut(const Mat &InputImageL, const Mat &InputImageR, vector<cv::KeyPoint> Vector_L,
                          vector<cv::KeyPoint> Vector_R) {
	Size size((InputImageL.cols + InputImageR.cols), max(InputImageL.rows, InputImageR.rows));
	cv::Mat OutputImage;
	// OutputImage.create(size, CV_MAKETYPE(InputImageL.depth(), 3));
	OutputImage.create(size, CV_8UC3);
	OutputImage = Scalar::all(0);
	Mat outImg_left = OutputImage(Rect(0, 0, InputImageL.cols, InputImageL.rows));
	Mat outImg_right = OutputImage(Rect(InputImageL.cols, 0, InputImageR.cols, InputImageR.rows));
	if (InputImageL.channels() == 1) {
		cvtColor(InputImageL, outImg_left, cv::COLOR_GRAY2BGR);
	}
	if (InputImageR.channels() == 1) {
		cvtColor(InputImageR, outImg_right, cv::COLOR_GRAY2BGR);
	}
	InputImageL.copyTo(outImg_left);
	InputImageR.copyTo(outImg_right);
	putText(OutputImage, to_string(Vector_L.size()), Point2f(50, 50), 3, 2, RandomColor(cv::getTickCount()));
	putText(OutputImage, to_string(Vector_R.size()), Point2f(InputImageL.cols + 50, 50), 3, 2,
	        RandomColor(cv::getTickCount()));
	
	for (int i = 0; i < min(Vector_L.size(), Vector_R.size()); i = i + 10) {
		Scalar colorNow = RandomColor(cv::getTickCount());
		circle(OutputImage, Vector_L[i].pt, 4, colorNow, 1, LINE_AA);
		circle(OutputImage, Point2f(Vector_R[i].pt.x + InputImageL.cols, Vector_R[i].pt.y), 4, colorNow, 1, LINE_AA);
		line(OutputImage, Vector_L[i].pt, Point2f(Vector_R[i].pt.x + InputImageL.cols, Vector_R[i].pt.y), colorNow, 1,
		     LINE_AA);
		
	}
	return OutputImage;
}


// 假设每个像素点的64个通道的数据都是按照顺序存储的
float *GetChannelData(float *p, int width, int height, int x, int y) {
	if (x >= 0 && x < width && y >= 0 && y < height) {
		return p + (y * width + x) * 64;
	} else {
		return nullptr; // 当输入的坐标超出范围时返回空指针
	}
}

bool compareKeyPoints(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2) {
	return kp1.response > kp2.response;
}

std::vector<cv::KeyPoint> sortFeaturePoints(std::vector<cv::KeyPoint> &featurePoints) {
	std::sort(featurePoints.begin(), featurePoints.end(), compareKeyPoints);
	return featurePoints;
}


int main() {
	vector<cv::KeyPoint> last_keypoint;
	Mat last_desc;
	Mat last_image;
	cv::Size size(640, 480);
	std::cout << size.width << size.height;
	const char *model_name = "../model/ALIKE.mnn";
	Net net = Net(model_name);
	cv::VideoCapture cap;
	cap.open("/home/moi/left_output_video.avi");
	// cap.open(0);
	if (!cap.isOpened()) {
		std::cerr << "ERROR!!Unable to open camera\n";
		return -1;
	}
	cv::Mat img;
	cv::Mat hot_pic(cv::Size(640, 480), CV_32FC1);
	while (true) {
		cap >> img;
		
		cv::Mat image;
		if (img.channels() == 3) {
			cvtColor(img, image, cv::COLOR_BGR2GRAY);
		} else {
			image = img;
		}
		if (image.empty()) {
			cout << "images input error! check please." << endl;
		}
		
		cv::resize(image, image, cv::Size(640, 480), 0, 0);
		TicToc a;
		net.Inference(image);
		auto hot_map = net.GetScoresValue();
		auto descriptors = net.GetDescriptorsValue();
		const auto *hotmap_index = (const float *) hot_map->buffer().host;
		int imag_w = hot_pic.size[1];
		hot_pic.forEach<float>([&](float &pixel, const int *position) {
			pixel = (hotmap_index[position[0] * imag_w + position[1]] * 255);
		});

#if USE_ORB
		std::vector<cv::KeyPoint> ssc_kp;
		Mat desc;
		mpIniORBextractor->extract_orb_fts(image,cv::Mat(),ssc_kp,desc);
#else
		std::vector<cv::KeyPoint> key_points;
		key_points.reserve(640 * 480);
		for (int i = 0; i < 480; i++) {
			for (int j = 0; j < 640; j++) {
				float hotmap_value = hotmap_index[i * 640 + j];
				if (hotmap_value > 1. / 255) {
					key_points.emplace_back(j, i, 7, -1, hotmap_value); // 将Point对象添加到key_points向量中
				}
			}
		}
		std::vector<cv::KeyPoint> sorted_feature_points = sortFeaturePoints(key_points);
		auto *descriptors_index = (float *) descriptors->buffer().host;
		TicToc b;
		vector<cv::KeyPoint> ssc_kp =
				// ssc(sorted_feature_points, 1000, 0.1, 640, 480);
				TopN(sorted_feature_points, 1000);
		Mat desc;
		for (const auto &kp: ssc_kp) {
			float *channel_data = GetChannelData(descriptors_index, 640, 480, kp.pt.x, kp.pt.y);
			cv::Mat des(1, 64, CV_32F, channel_data);
			cv::normalize(des, des, 1, cv::NORM_L2);
			desc.push_back(des);
		}
#endif
		cv::Mat out;
		cvtColor(image, out, cv::COLOR_GRAY2BGR);
		for (const cv::KeyPoint &kp: ssc_kp) {
			cv::circle(out, kp.pt, 1, cv::Scalar(0, 255, 0), -1); // 在图像上以绿色绘制半径为5的圆
		}
		if (!last_keypoint.empty() && !last_desc.empty()) {
#if USE_ORB
			cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING2);;
#else
			cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_L2);;
#endif
			std::vector<std::vector<cv::DMatch>> matches;
			matcher.knnMatch(last_desc, desc, matches, 2);
			std::vector<cv::DMatch> good_matchs;
			
			for (auto &matche: matches) {
				if (matche[0].distance < 0.7 * matche[1].distance) {
					good_matchs.push_back(matche[0]);
				}
			}
			std::vector<cv::KeyPoint> last_keypoints; // 存放上一帧的关键点
			std::vector<cv::KeyPoint> current_keypoints; // 存放当前帧的关键点
			for (const auto &good_match: good_matchs) {
				last_keypoints.push_back(last_keypoint[good_match.queryIdx]);
				current_keypoints.push_back(ssc_kp[good_match.trainIdx]);
			}
			std::cout << "good_matchs:" << good_matchs.size() << std::endl;
			// Mat matchimages = ConnectPointVectorOut(last_image, image, last_keypoints, current_keypoints);
			Mat matchimages;
			drawMatches(last_image, last_keypoint, image, ssc_kp, good_matchs, matchimages);
			imshow("matchImages", matchimages);
		}
		last_keypoint = ssc_kp;
		last_desc = desc;
		last_image = image;
		std::cout << " =======================" << a.toc() << std::endl;
		cv::imshow("hot_pic", hot_pic);
		cv::imshow("out", out);
		
		cv::waitKey(1);
	}
}