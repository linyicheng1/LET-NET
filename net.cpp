#include "net.hpp"
#include "common.hpp"
#include <fstream>
Net::Net(){};
Net::~Net(){};

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
	std::cout <<image.size() <<"  "<<image.channels() <<std::endl;
    cv::Mat preImage = image.clone();
    preImage.convertTo(preImage,CV_32FC3,1/255.);
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(preImage, bgrChannels); //bgrChannels会根据实际通道数自己调整。
    std::vector<float> chwImage;
    for (auto i = 0; i < bgrChannels.size(); i++)
    {  
        //HWC->CHW
        std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, preImage.cols * preImage.rows));
        chwImage.insert(chwImage.end(), data.begin(), data.end());
    }
    auto inTensor = net->getSessionInput(session, NULL);
	// inTensor->printShape();
    auto nchwTensor = shared_ptr<MNN::Tensor> (new MNN::Tensor(inTensor, MNN::Tensor::CAFFE));
    ::memcpy(nchwTensor->host<float>(), chwImage.data(), nchwTensor->elementSize() * 4);
    const float* hmap = (const float*)(nchwTensor->buffer().host);
    inTensor->copyFromHostTensor(nchwTensor.get());
}

void Net::Inference(const cv::Mat& image){
     Mat2Tensor(image);
     this->net->runSession(this->session);
 }

shared_ptr<MNN::Tensor> Net::GetScoresValue(){
    auto output= this->net->getSessionOutput(this->session, this->scoresOutName.c_str());
    auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputTensor.get());
    return outputTensor; 
}

shared_ptr<MNN::Tensor> Net::GetKeypointsValue(){
    auto output= this->net->getSessionOutput(this->session, this->keypointsOutName.c_str());
    auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputTensor.get());
    return outputTensor; 
}

shared_ptr<MNN::Tensor> Net::GetDescriptorsValue(){
    auto keypoints = this->GetKeypointsValue();
    auto scores =  this->GetScoresValue();
    int realDim = this->GetRealDim(scores->host<float>());
    this->realDim = realDim;
    auto output= this->net->getSessionOutput(this->session, this->descriptorsOutName.c_str());
    auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputTensor.get());
    std::vector<int> inputDims{1,256,realDim};
    auto descFinalTensor = (shared_ptr<MNN::Tensor>) MNN::Tensor::create<float>(inputDims, NULL, MNN::Tensor::CAFFE);
    auto descFinalOutput = sample_descriptors(keypoints->host<float>(),outputTensor->host<float>(),outputTensor->shape(),realDim,8);
    ::memcpy(descFinalTensor->host<float>(),descFinalOutput.get(),descFinalTensor->elementSize() * 4);
    return descFinalTensor;
}

// gao test
shared_ptr<MNN::Tensor> Net::GetDescriptorsValueOnly(){
     auto output= this->net->getSessionOutput(this->session, this->descriptorsOutName.c_str());
    auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputTensor.get());
    return outputTensor; 
}

shared_ptr<MNN::Tensor> Net::GetAllValue(){
    auto output= this->net->getSessionOutputAll(this->session);
    shared_ptr<MNN::Tensor> outputTensor;
    for(auto it:output){
        outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(it.second, MNN::Tensor::CAFFE));
        cout<<"it .first string: "<<it.first<<endl;
        cout<<"it second shape: "<<endl;
        // it.second->print();
        it.second->printShape();
    }
    // auto outputTensor = shared_ptr<MNN::Tensor>(new MNN::Tensor(output, MNN::Tensor::CAFFE));
    // output->copyToHostTensor(outputTensor.get());
    return outputTensor; 
}

int Net::GetRealDim(float* scores){
    return getRealDim(scores);
}