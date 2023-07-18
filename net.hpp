# pragma once
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<MNN/Interpreter.hpp>
#include<MNN/ImageProcess.hpp>
class Net{
    private:
        std::shared_ptr<MNN::Interpreter> net = nullptr;
        MNN::ScheduleConfig config;
        MNN::Session *session = nullptr;
        MNN::BackendConfig backendConfig;
        /*MNN 后端配置*/
        std::string scoresOutName = "score";
        std::string keypointsOutName = "keypoints";
        std::string descriptorsOutName = "descriptor";
    public:
        Net();
        explicit Net(const char* modelPath);
        void Mat2Tensor(const cv::Mat& image);
        void Inference(const cv::Mat& image);
        std::shared_ptr<MNN::Tensor> GetScoresValue();
        std::shared_ptr<MNN::Tensor> GetDescriptorsValueOnly();
        ~Net();
};