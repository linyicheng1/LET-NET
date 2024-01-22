#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace orb_exteactor {
	class ExtractorNode {
	public:
		ExtractorNode() : bNoMore(false) {}
		
		void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);
		
		std::vector<cv::KeyPoint> vKeys;
		cv::Point2i UL, UR, BL, BR;
		std::list<ExtractorNode>::iterator lit;
		bool bNoMore;
	};
	
	
	class CV_EXPORTS_W ORBextractor {
	public:
		CV_WRAP ORBextractor(int nfeatures, float scaleFactor, int nlevels,
		                     int iniThFAST, int minThFAST);
		
		~ORBextractor() {}
		
		// Compute the ORB features and descriptors on an image.
		// ORB are dispersed on the image using an octree.
		// Mask is ignored in the current implementation.
		CV_WRAP void extract_orb_fts(InputArray image, InputArray mask, vector<cv::KeyPoint> &kps,
		                             OutputArray descriptors);
		
		int inline GetLevels() {
			return nlevels;
		}
		
		float inline GetScaleFactor() {
			return scaleFactor;
		}
		
		std::vector<float> inline GetScaleFactors() {
			return mvScaleFactor;
		}
		
		std::vector<float> inline GetInverseScaleFactors() {
			return mvInvScaleFactor;
		}
		
		std::vector<float> inline GetScaleSigmaSquares() {
			return mvLevelSigma2;
		}
		
		std::vector<float> inline GetInverseScaleSigmaSquares() {
			return mvInvLevelSigma2;
		}
		
		std::vector<cv::Mat> mvImagePyramid;
	
	protected:
		
		void ComputePyramid(cv::Mat image);
		
		void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> > &allKeypoints);
		
		std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
		                                            const int &maxX, const int &minY, const int &maxY,
		                                            const int &nFeatures, const int &level);
		
		void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> > &allKeypoints);
		
		std::vector<cv::Point> pattern;
		
		int nfeatures;
		double scaleFactor;
		int nlevels;
		int iniThFAST;
		int minThFAST;
		
		std::vector<int> mnFeaturesPerLevel;
		
		std::vector<int> umax;
		
		std::vector<float> mvScaleFactor;
		std::vector<float> mvInvScaleFactor;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;
	};
}

#endif

