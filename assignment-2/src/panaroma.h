#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>

#include "utils.hpp"

class Panaroma
{
	public:
		Panaroma(std::vector<cv::Mat> images);

		void generateMatches(const cv::Mat &img1, const cv::Mat &img2,
            Eigen::MatrixXf &X1, Eigen::MatrixXf &X2);

		void sampleFromX1X2(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2,
			const std::vector<int> &sample_indices, Eigen::MatrixXf &sample_X1,
		    Eigen::MatrixXf &sample_X2);

       	void estimateHomography(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2,
            Eigen::MatrixXf &H);

        void estimateRansacHomography(const Eigen::MatrixXf &X1,
            const Eigen::MatrixXf &X2, const float &dist_threshold,
            const float &ratio_threshold, Eigen::MatrixXf &H, std::vector<int> &inlier_indices);

		void warpImage(const cv::Mat &src_img, const Eigen::Matrix3f &H, cv::Mat &dst_img);

        void stitchImages();

		void run(const float &dist_threshold = 100, const float &ratio_threshold = 0.5);

	private:
		std::vector<cv::Mat> images;
};
