#include "common.hpp"

float safeGet(float* values,int x,int y, int n,int c,int C,int H,int W){
    float value = 0.;
    if(x >= 0 && x < W && y >= 0 && y < H)
        return values[(((n * C)+c)*H + y) * W + x];
    return value;
}

int getRealDim(float* hostValue){
    for(int i = 1; i < 5000; i++){
        if(hostValue[i] == hostValue[0]){
            if(i < 4999 && hostValue[i+1] == hostValue[0])
                return i;
        }
    }
    return 5000;
}

shared_ptr<float> grid_sample(float* keypoints,float* descriptors,int N,int C,int IH,int IW,int H,int W){
    auto output = shared_ptr<float> (new float[1*C*1*W]);
    for(int n = 0; n < N; n++){
        for(int h = 0; h < H; h++){
            for(int w = 0; w < W; w++){
                float ix = keypoints[2*w];
                float iy = keypoints[2*w+1];
                ix = ((ix + 1) / 2.) * (IW - 1);
                iy = ((iy + 1) / 2.) * (IH - 1);
                int ix_nw = floor(ix);
                int iy_nw = floor(iy);
                int ix_ne = ix_nw + 1;
                int iy_ne = iy_nw;
                int ix_sw = ix_nw;
                int iy_sw = iy_nw + 1;
                int ix_se = ix_nw + 1;
                int iy_se = iy_nw + 1;
                float nw = (ix_se - ix)    * (iy_se - iy);
                float ne = (ix    - ix_sw) * (iy_sw - iy);
                float sw = (ix_ne - ix)    * (iy    - iy_ne);
                float se = (ix    - ix_nw) * (iy    - iy_nw);
                for(int c = 0; c < C; c++){
                    float nw_val = safeGet(descriptors, ix_nw, iy_nw, n, c, C,IH, IW);
                    float ne_val = safeGet(descriptors, ix_ne, iy_ne, n, c, C,IH, IW);
                    float sw_val = safeGet(descriptors, ix_sw, iy_sw, n, c, C,IH, IW);
                    float se_val = safeGet(descriptors, ix_se, iy_se, n, c, C,IH, IW);
                    float out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
                    output.get()[(((n * C)+c)*H + h) * W + w] = out_val;
                }
            }
        }
    }
    return output;
}

shared_ptr<float> sample_descriptors(float* keypoints,float* descriptors,vector<int> shape, int realDim, int s){
    int N = shape[0];
    int C = shape[1];
    int IH = shape[2];
    int IW = shape[3];
    int H = 1;
    int W = realDim;
    //normalize keypointes to (-1,1)
    float* tempKeypoints = new float[realDim*2];  
    memcpy(tempKeypoints,keypoints,realDim * 2 * 4);
    for(int i = 0; i < realDim; i++){
        tempKeypoints[2*i] = tempKeypoints[2*i] - s / 2. + .5;
        tempKeypoints[2*i + 1] = tempKeypoints[2*i + 1] - s / 2. + .5;
        tempKeypoints[2*i] = tempKeypoints[2*i] / (IW*s - s/ 2. - 0.5) * 2. - 1.;
        tempKeypoints[2*i + 1] = tempKeypoints[2*i + 1] / (IH*s - s/ 2. - 0.5) * 2. - 1.;
    }
    // cout<<"tempKeypoints : ";
    // for(int i=0;i<20;++i){
    //     cout<<tempKeypoints[i]<<" ";
    // }
    // * checked ,截至到这里，数据都没出啥大问题
    // cout<<"descriptors : ";
    // for(int i=0;i<100;++i){
    //     cout<<descriptors[i]<<" ";
    //     if((i+1)%5 == 0){
    //         cout<<endl;
    //     }
    // }
    //tempKeypoints [1xrealDim*2] =[realDim x2] descriptors[1 x (256, 60, 80)] = [256x60x80]
    // N=1,C=256,IH=60,IW=80,H=1,W=2456=realDim
    auto origin = grid_sample(tempKeypoints,descriptors,N,C,IH,IW,H,W);
    //reshape [c*w]->[w*c] 
    auto output = shared_ptr<float> (new float[C*W]);
    // cout<<"test : "<<orgin.get()[0]<<endl;
    float* oriData = origin.get();
    float* outData = output.get();
    for(int i =0;i<W;++i){
        for(int j =0;j<C;++j){
            outData[i*C+j]=oriData[j*W+i];
        }
    }
    delete [] tempKeypoints;
    return output;
}

//简化点数目
shared_ptr<float> simple_nms(const float* hotmapIndex , vector<int> shape, int realDim){
    auto output = shared_ptr<float>(new float[realDim]);
    float* outputDate = output.get();
    std::fill_n(outputDate, realDim, 0.);
    int nms_radius = 4;
    bool *max_mask = new bool[realDim];
    auto s1 = max_pool(hotmapIndex,shape,realDim,nms_radius);
    float *s1data = s1.get();
    for(int i = 0;i<realDim;++i){
        max_mask[i] = s1data[i]==hotmapIndex[i]?true:false;
        
    }
    // cout<<endl;
    bool *supp_mask = new bool[realDim];
    float *supp_scores = new float[realDim];
    bool *new_max_mask = new bool[realDim];
    for(int i = 0;i<2;++i){
        auto supp_mask1 = max_pool(max_mask,shape, realDim,nms_radius);
        bool *supp_maskData = supp_mask1.get();
        // bool *supp_mask = new bool[realDim];
        // float *supp_scores = new float[realDim];
        for(int i = 0;i<realDim;++i){
            supp_mask[i] = supp_maskData[i]>0?true:false;
            supp_scores[i] = supp_maskData[i]>0?0.:hotmapIndex[i];
        }
        auto new_max_mask1 = max_pool(supp_scores,shape, realDim,nms_radius);
        float *new_max_maskData = new_max_mask1.get();
        // bool *new_max_mask = new bool[realDim];
        for(int i = 0;i<realDim;++i){
            new_max_mask[i] = supp_scores[i]==new_max_maskData[i]?true:false;
            max_mask[i] = max_mask[i] | (new_max_mask[i]& (!supp_mask[i]));
        }
    }
    for(int i = 0;i<realDim;++i){
        outputDate[i] = max_mask[i]?hotmapIndex[i]:0.;
    }
    delete [] supp_mask;
    delete [] supp_scores;
    delete [] new_max_mask;
    delete [] max_mask;
    
    return output;
}

/***
 * 实现max_pool池化操作，但精简版，默认步长stride = 1。
 * kernel_size = nms_radius*2+1 = 9 ， stride=1, padding=4
    // 理解，实际上执行了池化操作，就是在9x9格的矩阵，选最大值作为当前矩阵max结果。
    // stride=1指每次移动一格，
    // nms_radius = padding=4指整个矩阵周围补充4格（这个值不会被加入选择结果中。）
 * @param [in] 浮点数组指针
 * @param [in] 数组形状
 * @param 数组元素个数
 * @param 池半径（实际范围nms_radius*2+1 ^2）
*/
template<typename T>
shared_ptr<T> max_pool(const T *hotmapIndex, vector<int> feature, int realDim, int nms_radius)
{
    int row = feature[0], clo = feature[1];
    auto output = shared_ptr<T>(new T[realDim]);
    T *outData = output.get();
    vector<vector<T>> feaMap(row+2*nms_radius,vector<T>(clo+2*nms_radius,FLT_MIN));
    for (int r = nms_radius; r < (row + nms_radius); ++r)
    {
        for (int c = nms_radius; c < (clo + nms_radius); ++c)
        {
            feaMap[r][c] = hotmapIndex[(r-nms_radius) * clo + c-nms_radius];
            // cout<<setiosflags(ios::fixed)<<setprecision(5)<<hotmapIndex[r * row + c]<<" ";
        }
    }

    for (int r = nms_radius; r < (row+nms_radius); ++r)
    {
        for (int c = nms_radius; c < (clo+nms_radius); ++c)
        {
            // kernel size内取max
            T max = feaMap[r][c];
            for (int i = r - nms_radius; i < r + nms_radius; ++i)
            {
                for (int j = c - nms_radius; j < c + nms_radius; ++j)
                {
                    // max = std::max(max, feaMap[i][j]); //不能比较bool?
                    T a = feaMap[i][j];
                    max = max>a?max:a;
                }
            }
            outData[(r-nms_radius) * clo + c-nms_radius] = max;
        }
    }
    return output;
}
