#include "tracking.h"

void corner_tracking::update(const cv::Mat& score, const cv::Mat& desc) {

    if (trackedPoints.empty()) {// first frame
        trackedPoints = extractFeature(score);
        trackedPointsHistory.resize(trackedPoints.size());
        for (size_t i = 0; i < trackedPoints.size(); i++) {
            trackedPointsHistory[i].push_back(trackedPoints[i]);
        }
    } else {
        std::vector<cv::Point2f> trackedPointsNew;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(
                prevDesc,
                desc,
                trackedPoints,
                trackedPointsNew,
                status,
                err);

        std::vector<cv::Point2f> tracked = {};
        std::vector<std::vector<cv::Point2f>> trackedHistory = {};
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                tracked.push_back(trackedPointsNew[i]);
                trackedPointsHistory[i].push_back(trackedPointsNew[i]);
                if (trackedPointsHistory[i].size() > 5) {
                    trackedPointsHistory[i].erase(trackedPointsHistory[i].begin());
                }
                trackedHistory.push_back(trackedPointsHistory[i]);
            }
        }
        std::vector<cv::Point2f> add = extractFeature(score, 20, tracked);
        std::vector<std::vector<cv::Point2f>> add_history(add.size());
        for (size_t i = 0; i < add.size(); i++) {
            add_history[i].push_back(add[i]);
        }

        trackedPoints.clear();
        trackedPointsHistory.clear();

        trackedPoints.insert(trackedPoints.end(), tracked.begin(), tracked.end());
        trackedPoints.insert(trackedPoints.end(), add.begin(), add.end());
        trackedPointsHistory.insert(trackedPointsHistory.end(), trackedHistory.begin(), trackedHistory.end());
        trackedPointsHistory.insert(trackedPointsHistory.end(), add_history.begin(), add_history.end());
    }
    prevDesc = desc;
}

void corner_tracking::show(cv::Mat &img) {
    for (auto& p : trackedPoints) {
        cv::circle(img, p, 2, cv::Scalar(0, 255, 0), -1);
    }
    for (auto& history : trackedPointsHistory) {
        for (size_t i = 1; i < history.size(); i++) {
            cv::line(img, history[i - 1], history[i], cv::Scalar(0, 0, 255), 1);
        }
    }
    cv::imshow("tracking", img);
}

std::vector<cv::Point2f> corner_tracking::extractFeature(
        const cv::Mat& score,
        int ncellsize,
        const std::vector<cv::Point2f>& vcurkps)
{
    if (score.empty()) {
        return std::vector<cv::Point2f>();
    }

    size_t ncols = score.cols;
    size_t nrows = score.rows;

    size_t nhalfcell = ncellsize / 4;

    size_t nhcells = nrows / ncellsize;
    size_t nwcells = ncols / ncellsize;
    size_t nbcells = nhcells * nwcells;

    std::vector<cv::Point2f> vdetectedpx;
    vdetectedpx.reserve(nbcells);

    std::vector<std::vector<bool>> voccupcells(
            nhcells + 1,
            std::vector<bool>(nwcells + 1, false)
    );

    cv::Mat mask = cv::Mat::ones(score.rows, score.cols, CV_8UC1);

    for (const auto& px : vcurkps) {
        voccupcells[px.y / ncellsize][px.x / ncellsize] = true;
        cv::circle(mask, px, nhalfcell, cv::Scalar(0.), -1);
    }

    size_t nboccup = 0;

    std::vector<std::vector<cv::Point2f>> vvdetectedpx(nbcells);
    std::vector<std::vector<cv::Point2f>> vvsecdetectionspx(nbcells);

    auto cvrange = cv::Range(0, nbcells);

    parallel_for_(cvrange, [&](const cv::Range& range)
    {
        for (int i = range.start; i < range.end; i ++) {

            size_t r = floor(i / nwcells);
            size_t c = i % nwcells;

            if( voccupcells[r][c] ) {
                nboccup++;
                continue;
            }

            size_t x = c*ncellsize;
            size_t y = r*ncellsize;

            cv::Rect hroi(x,y,ncellsize,ncellsize);

            if( x+ncellsize < ncols-1 && y+ncellsize < nrows-1 ) {

                double dminval, dmaxval;
                cv::Point minpx, maxpx;

                cv::minMaxLoc(score(hroi).mul(mask(hroi)), &dminval, &dmaxval, &minpx, &maxpx);
                maxpx.x += x;
                maxpx.y += y;

                if( dmaxval >= 0.2) {
                    vvdetectedpx.at(i).push_back(maxpx);
                    cv::circle(mask, maxpx, nhalfcell, cv::Scalar(0.), -1);
                }

                cv::minMaxLoc(score(hroi).mul(mask(hroi)), &dminval, &dmaxval, &minpx, &maxpx);
                maxpx.x += x;
                maxpx.y += y;

                if( dmaxval >= 0.2)
                {
                    vvsecdetectionspx.at(i).push_back(maxpx);
                    cv::circle(mask, maxpx, nhalfcell, cv::Scalar(0.), -1);
                }
            }
        }
    });

    for (const auto& vpx:vvdetectedpx) {
        if (!vpx.empty()) {
            vdetectedpx.insert(vdetectedpx.end(), vpx.begin(), vpx.end());
        }
    }

    size_t nbkps = vdetectedpx.size();

    if (nbkps + nboccup < nbcells) {
        size_t nbsec = nbcells - nbkps - nboccup;
        size_t k = 0;
        for (const auto &vseckp : vvsecdetectionspx) {
            if (!vseckp.empty()) {
                vdetectedpx.push_back(vseckp.back());
                k ++;
                if (k == nbsec) {
                    break;
                }
            }
        }
    }

    return vdetectedpx;
}



