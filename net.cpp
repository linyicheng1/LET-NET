#include "net.hpp"
#include <memory>
Net::Net(){};
Net::~Net()= default;

Net::Net(const char* modelPath){
    this->net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath));
    this->backendConfig.precision =MNN::BackendConfig::Precision_High;
    this->backendConfig.power = MNN::BackendConfig::Power_Normal;
    this->backendConfig.memory = MNN::BackendConfig::Memory_High;
    this->config.backendConfig = & this->backendConfig;
	this->config.mode = MNN_GPU_TUNING_NORMAL | MNN_GPU_MEMORY_IMAGE;
    this->config.type = MNN_FORWARD_OPENCL;
    this->session = this->net->createSession(this->config);
}
void Net::Mat2Tensor(const cv::Mat& image){
    cv::Mat pre_image = image.clone();
    pre_image.convertTo(pre_image,CV_32FC3,1/255.);
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(pre_image, bgrChannels);
    std::vector<float> chwImage;
    for (auto i = 0; i < bgrChannels.size(); i++)
    {  
        //HWC->CHW
        std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, pre_image.cols * pre_image.rows));
        chwImage.insert(chwImage.end(), data.begin(), data.end());
    }
    auto in_tensor = net->getSessionInput(session, NULL);;
    auto nchw_tensor = std::make_shared<MNN::Tensor> (in_tensor, MNN::Tensor::CAFFE);
    ::memcpy(nchw_tensor->host<float>(), chwImage.data(), nchw_tensor->elementSize() * 4);
    // const auto* hmap = (const float*)(nchw_tensor->buffer().host);
    in_tensor->copyFromHostTensor(nchw_tensor.get());
}
void Net::Inference(const cv::Mat& image){
     Mat2Tensor(image);
     this->net->runSession(this->session);
 }
std::shared_ptr<MNN::Tensor> Net::GetScoresValue(){
    auto output= this->net->getSessionOutput(this->session, this->scoresOutName.c_str());
    auto output_tensor = std::make_shared<MNN::Tensor>(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(output_tensor.get());
    return output_tensor;
}
std::shared_ptr<MNN::Tensor> Net::GetDescriptorsValueOnly(){
     auto output= this->net->getSessionOutput(this->session, this->descriptorsOutName.c_str());
    auto output_tensor = std::make_shared<MNN::Tensor>(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(output_tensor.get());
    return output_tensor;
}