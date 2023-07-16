#ifndef __TRACKING_H_
#define __TRACKING_H_
#include <vector>
#include "opencv2/opencv.hpp"


class corner_tracking
{
public:
    corner_tracking() = default;
    ~corner_tracking() = default;
    void update(const cv::Mat& score, const cv::Mat& desc);
    void show(cv::Mat& img);
private:
    std::vector<cv::Point2f> extractFeature(
            const cv::Mat& score,
            int ncellsize = 20,
            const std::vector<cv::Point2f>& points = std::vector<cv::Point2f>());

    std::vector<cv::Point2f> trackedPoints;
    std::vector<cv::Point2f> prevTrackedPoints;
    cv::Mat prevScore;
    cv::Mat prevDesc;
    std::vector<std::vector<cv::Point2f>> trackedPointsHistory;
};

#endif //__TRACKING_H_
