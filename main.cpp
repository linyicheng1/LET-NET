#include "net.h"
#include "mat.h"
#include "opencv2/opencv.hpp"
#include "chrono"
#include "tracking.h"

#define IMG_H 240
#define IMG_W 320

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cout<<" Usage: ./demo <model_param> <model_bin> <video_path> <video_path> or ./demo <model_param> <model_bin> <image_path_1> <image_path_2>"<<std::endl;
        return -1;
    }
    cv::VideoCapture capture;
    cv::VideoWriter writer("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(320, 240));
    cv::Mat img1, img2;
    bool is_video = false;
    if (argc == 4) {
        is_video = true;
        std::string video_path = argv[3];
        capture.open(video_path);
        if (!capture.isOpened()) {
            std::cout<<" Error opening video file !"<<std::endl;
            return -1;
        }
    } else {
        std::string img_path_1 = argv[3];
        std::string img_path_2 = argv[4];
        img1 = cv::imread(img_path_1);
        img2 = cv::imread(img_path_2);
        cv::resize(img1, img1, cv::Size(IMG_W, IMG_H));
        cv::resize(img2, img2, cv::Size(IMG_W, IMG_H));
        if (img1.empty() || img2.empty()) {
            std::cout << " Error opening image file !" << std::endl;
            return -1;
        }
    }

    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
    const float mean_vals_inv[3] = {0, 0, 0};
    const float norm_vals_inv[3] = {255.f, 255.f, 255.f};

    ncnn::Net net;
    net.load_param(argv[1]);
    net.load_model(argv[2]);

    cv::Mat score(IMG_H, IMG_W, CV_8UC1);
    cv::Mat desc(IMG_H, IMG_W, CV_8UC3);
    cv::Mat frame;
    ncnn::Mat in;
    ncnn::Mat out1, out2;

    corner_tracking tracker;
    while (true) {

        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(1);

        if (is_video) {
            capture >> frame;
        } else {
            static int i = 0;
            if (i == 0)
                frame = img1;
            else if (i == 1)
                frame = img2;
            else
                break;
            i++;
        }

        if (frame.empty())
            break;

        cv::resize(frame, frame, cv::Size(IMG_W, IMG_H));

        //////////////////////////  opencv image to ncnn mat  //////////////////////////
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        in = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows);
        in.substract_mean_normalize(mean_vals, norm_vals);

        //////////////////////////  ncnn forward  //////////////////////////

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

        ex.input("input", in);
        ex.extract("score", out1);
        ex.extract("descriptor", out2);

        //////////////////////////  ncnn mat to opencv image  //////////////////////////

        std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
        out1.substract_mean_normalize(mean_vals_inv, norm_vals_inv);
        out2.substract_mean_normalize(mean_vals_inv, norm_vals_inv);

//        memcpy((uchar*)score.data, out1.data, sizeof(float) * out1.w * out1.h);
        out1.to_pixels(score.data, ncnn::Mat::PIXEL_GRAY);
        out2.to_pixels(desc.data, ncnn::Mat::PIXEL_BGR);

        std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();


        //////////////////////////  show times  //////////////////////////

        std::chrono::duration<double> time_used_1 = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        std::chrono::duration<double> time_used_2 = std::chrono::duration_cast<std::chrono::duration<double>>(t3-t2);
        std::chrono::duration<double> time_used_3 = std::chrono::duration_cast<std::chrono::duration<double>>(t4-t3);

        std::cout<<"time_used 1 : "<<time_used_1.count()*1000<<"ms"<<std::endl;
        std::cout<<"time_used 2 : "<<time_used_2.count()*1000<<"ms"<<std::endl;
        std::cout<<"time_used 3 : "<<time_used_3.count()*1000<<"ms"<<std::endl;

        //////////////////////////  show result  //////////////////////////
        cv::Mat new_desc = desc.clone();
        tracker.update(score, new_desc);
        tracker.show(frame);
        if (is_video) {
            writer << frame;
        }
        cv::waitKey(500);
    }
    writer.release();
    capture.release();
    return 0;
}
