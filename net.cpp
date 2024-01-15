#include "net.hpp"
#include <memory>

Net::Net()= default;
Net::~Net()= default;

Net::Net(const char* modelPath){
    this->net_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelPath));
    this->backend_config_.precision =MNN::BackendConfig::Precision_High;
    this->backend_config_.power = MNN::BackendConfig::Power_Normal;
    this->backend_config_.memory = MNN::BackendConfig::Memory_Normal;
    this->config_.backendConfig = & this->backend_config_;
	this->config_.mode = MNN_GPU_TUNING_FAST | MNN_GPU_MEMORY_IMAGE;
    this->config_.type = MNN_FORWARD_OPENCL;
    this->session_ = this->net_->createSession(this->config_);
}
void Net::Mat2Tensor(const cv::Mat& image){
	cv::Mat pre_image;
	std::vector<float> chw_image;
	if(image.channels() == 3) {
		image.convertTo(pre_image, CV_32FC3, 1/255.0);
		std::vector<cv::Mat> bgr(3);
		cv::split(pre_image, bgr);
		chw_image.reserve(pre_image.cols * pre_image.rows * 3);
		for (const auto &item: bgr) {
			const auto *data = item.ptr<float>();
			chw_image.insert(chw_image.end(), data, data + item.total());
		}
	}
	if(image.channels() == 1){
		image.convertTo(pre_image, CV_32FC1, 1/255.0);
		chw_image.reserve(pre_image.cols * pre_image.rows);
		const auto *data = pre_image.ptr<float>();
		chw_image.insert(chw_image.end(), data, data + pre_image.total());
	}
	auto in_tensor = net_->getSessionInput(session_, nullptr);
	auto nchw_tensor = std::make_shared<MNN::Tensor>(in_tensor, MNN::Tensor::CAFFE);
	::memcpy(nchw_tensor->host<float>(), chw_image.data(), chw_image.size() * sizeof(float));
    in_tensor->copyFromHostTensor(nchw_tensor.get());
}
void Net::Inference(const cv::Mat& image){
     Mat2Tensor(image);
     this->net_->runSession(this->session_);
 }
std::shared_ptr<MNN::Tensor> Net::GetScoresValue(){
    auto output= this->net_->getSessionOutput(this->session_, this->scores_out_name_.c_str());
    auto output_tensor = std::make_shared<MNN::Tensor>(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(output_tensor.get());
    return output_tensor;
}
std::shared_ptr<MNN::Tensor> Net::GetDescriptorsValue(){
     auto output= this->net_->getSessionOutput(this->session_, this->descriptors_out_name_.c_str());
    auto output_tensor = std::make_shared<MNN::Tensor>(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(output_tensor.get());
    return output_tensor;
}