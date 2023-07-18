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
        const char* modelPath;
        MNN::ScheduleConfig config;
        MNN::Session *session = nullptr;
        MNN::BackendConfig backendConfig;
        /*MNN 后端配置*/
        std::string scoresOutName = "score";
        std::string keypointsOutName = "keypoints";
        std::string descriptorsOutName = "descriptor";
    public:
        Net();
        //初始化模型
        Net(const char* modelPath);
        //opencv Mat转 MNN Tensor
        void Mat2Tensor(const cv::Mat& image);
        void Inference(const cv::Mat& image);
        //获取scores的值{1,readDim}
        std::shared_ptr<MNN::Tensor> GetScoresValue();
        //获取kpts的值{1,readDim,2}
        std::shared_ptr<MNN::Tensor> GetKeypointsValue();
        //获取descr的值{1,256,readDim}
        std::shared_ptr<MNN::Tensor> GetDescriptorsValue();
        // gao 获取所有值？
        std::shared_ptr<MNN::Tensor> GetAllValue();
        std::shared_ptr<MNN::Tensor> GetDescriptorsValueOnly();
        //获取输出的实际维度值
        int GetRealDim(float* scores);

        int realDim = -1;
        ~Net();
};