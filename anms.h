#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#if CV_MAJOR_VERSION >= 4
#include "opencv2/imgcodecs/legacy/constants_c.h"
#endif
#include <cstdlib>

using namespace std;
vector<cv::KeyPoint> ssc(vector<cv::KeyPoint> keyPoints, int numRetPoints,
                         float tolerance, int cols, int rows) {
  // several temp expression variables to simplify solution equation
  int exp1 = rows + cols + 2 * numRetPoints;
  long long exp2 =
      ((long long)4 * cols + (long long)4 * numRetPoints +
       (long long)4 * rows * numRetPoints + (long long)rows * rows +
       (long long)cols * cols - (long long)2 * rows * cols +
       (long long)4 * rows * cols * numRetPoints);
  double exp3 = sqrt(exp2);
  double exp4 = numRetPoints - 1;

  double sol1 = -round((exp1 + exp3) / exp4); // first solution
  double sol2 = -round((exp1 - exp3) / exp4); // second solution

  // binary search range initialization with positive solution
  int high = (sol1 > sol2) ? sol1 : sol2;
  int low = floor(sqrt((double)keyPoints.size() / numRetPoints));
  low = max(1, low);

  int width;
  int prevWidth = -1;

  vector<int> ResultVec;
  bool complete = false;
  unsigned int K = numRetPoints;
  unsigned int Kmin = round(K - (K * tolerance));
  unsigned int Kmax = round(K + (K * tolerance));

  vector<int> result;
  result.reserve(keyPoints.size());
  while (!complete) {
    width = low + (high - low) / 2;
    if (width == prevWidth ||
        low >
            high) { // needed to reassure the same radius is not repeated again
      ResultVec = result; // return the keypoints from the previous iteration
      break;
    }
    result.clear();
    double c = (double)width / 2.0; // initializing Grid
    int numCellCols = floor(cols / c);
    int numCellRows = floor(rows / c);
    vector<vector<bool>> coveredVec(numCellRows + 1,
                                    vector<bool>(numCellCols + 1, false));

    for (unsigned int i = 0; i < keyPoints.size(); ++i) {
      int row =
          floor(keyPoints[i].pt.y /
                c); // get position of the cell current point is located at
      int col = floor(keyPoints[i].pt.x / c);
      if (coveredVec[row][col] == false) { // if the cell is not covered
        result.push_back(i);
        int rowMin = ((row - floor(width / c)) >= 0)
                         ? (row - floor(width / c))
                         : 0; // get range which current radius is covering
        int rowMax = ((row + floor(width / c)) <= numCellRows)
                         ? (row + floor(width / c))
                         : numCellRows;
        int colMin =
            ((col - floor(width / c)) >= 0) ? (col - floor(width / c)) : 0;
        int colMax = ((col + floor(width / c)) <= numCellCols)
                         ? (col + floor(width / c))
                         : numCellCols;
        for (int rowToCov = rowMin; rowToCov <= rowMax; ++rowToCov) {
          for (int colToCov = colMin; colToCov <= colMax; ++colToCov) {
            if (!coveredVec[rowToCov][colToCov])
              coveredVec[rowToCov][colToCov] =
                  true; // cover cells within the square bounding box with width
                        // w
          }
        }
      }
    }

    if (result.size() >= Kmin && result.size() <= Kmax) { // solution found
      ResultVec = result;
      complete = true;
    } else if (result.size() < Kmin)
      high = width - 1; // update binary search range
    else
      low = width + 1;
    prevWidth = width;
  }
  // retrieve final keypoints
  vector<cv::KeyPoint> kp;
  kp.reserve(ResultVec.size());
  for (int i : ResultVec)
    kp.push_back(keyPoints[i]);

  return kp;
}
