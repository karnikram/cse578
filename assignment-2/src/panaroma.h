#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>

#include "utils.h"
#include <string>

/** 
 * Class that implements image mosaicing.
 */

class Panaroma
{
	public:
		/** Constructor that saves the input sequence of images, and output file path. */
		Panaroma(const std::vector<cv::Mat> &images, const std::string &output_path);
        
		/** Generates pairs of corresponding points from img1, img2. */
		void generateMatches(const cv::Mat &img1, const cv::Mat &img2, 
				Eigen::MatrixXf &X1, Eigen::MatrixXf &X2, const int &id);

		/** Samples pairs of corresponding points from X1, X2 according to sample_indices. */
		void sampleFromX1X2(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2, 
				const std::vector<int> &sample_indices, Eigen::MatrixXf &sample_X1, 
				Eigen::MatrixXf &sample_X2);

		/** H estimation using DLT.
		 * Called from within estimateRansacHomography
		 */
		void estimateHomography(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2, 
				Eigen::MatrixXf &H);
        
		/** Calculates reprojection error (squared norm) between x2 and x1 after projection by H. */
		float calcReprojectionError(const Eigen::MatrixXf &x1, const Eigen::MatrixXf &x2, const Eigen::MatrixXf &H);

		/** Calculates average reprojection error over all pairs of corresponding points in X1,X2. */
		float calcAvgReprojectionError(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2, const Eigen::MatrixXf &H);
        
		/** Estimates H witihin a ransac scheme.
		 * \param X1 Keypoints from img1
		 * \param X2 Corresponding keypoints from img2
		 * \param dist_threshold The distance threshold (squared norm) to determine inlier for ransac
		 * \param ratio_threshold The ratio of inliers for ransac
		 * \param H The estimated H matrix
		 * \param inlier_indices The indices of inlier points used to estimate H
		 */
		void estimateRansacHomography(const Eigen::MatrixXf &X1,
			const Eigen::MatrixXf &X2, const float &dist_threshold,
			const float &ratio_threshold, Eigen::MatrixXf &H, std::vector<int> &inlier_indices);

		/** Warps src_img using H (inverse) and stores result in warped_img. */
		void warpImage(cv::Mat src_img, const Eigen::Matrix3f &H, cv::Mat &warped_img);

		/** Aligns img2 with img1, and stores in result. */
		void stitch(cv::Mat img1, cv::Mat img2, cv::Mat &result);

		/** Runs the entire image mosaicing procedure.
		 * \param dist_threshold The distance threshold (squred norm) to determine inlier for ransac
		 * \param ratio_threshold The ratio of inliers for ransac
		 */
		void run(const float &dist_threshold = 10, const float &ratio_threshold = 0.5);

	private:
		std::vector<cv::Mat> images;
		std::string output_path;
};
