# pragma once
#include<vector>
#include <iostream>
#include<math.h>
#include<string.h>
#include<memory>

#include<float.h>
#include<iomanip>

using namespace std;

 /*获取Tensor的索引值，x=>w,y=>h,n=>b,c=>C*/
 float safeGet(float* values,int x,int y, int n,int c,int C,int H,int W);
 
 /*获取输出的实际维度，由于MNN输出不能是可变的，所以就固定了一个大的维度值(5000),
 发现超过实际维度的值跟第0个值保持一样，利用该特性获取实际维度*/
 int getRealDim(float* hostValue);

 /*torch.grid_sample的C++实现，仅实现双线性插值*/
shared_ptr<float> grid_sample(float* keypoints,float* descriptors,int N,int C,int IH,int IW,int H,int W);

 /*superpoint.py sample_descriptors函数C++实现*/
shared_ptr<float> sample_descriptors(float* keypoints,float* descriptors,vector<int> shape, int realDim, int s);

 /*maxpool函数C++实现*/
template<typename T>
shared_ptr<T> max_pool(const T *hotmapIndex, vector<int> feature, int realDim, int nms_radius);

shared_ptr<float> simple_nms(const float* hotmapIndex , vector<int> shape, int realDim);