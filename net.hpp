# pragma once
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<MNN/Interpreter.hpp>
#include<MNN/ImageProcess.hpp>
class Net{
    private:
        std::shared_ptr<MNN::Interpreter> net_ = nullptr;
        MNN::ScheduleConfig config_;
        MNN::Session *session_ = nullptr;
        MNN::BackendConfig backend_config_;
        /*MNN 后端配置*/
        std::string scores_out_name_ = "score";
        std::string descriptors_out_name_ = "descriptor";
    public:
        Net();
        explicit Net(const char* modelPath);
        void Mat2Tensor(const cv::Mat& image);
        void Inference(const cv::Mat& image);
        std::shared_ptr<MNN::Tensor> GetScoresValue();
        std::shared_ptr<MNN::Tensor> GetDescriptorsValue();
        ~Net();
	int realDim = -1;
	//获取输出的实际维度值
	int GetRealDim(float* scores);
};