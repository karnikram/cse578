#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>

class StereoMatcher
{
	public:
		StereoMatcher(const std::vector<cv::Mat> &stereo_pair, const std::string &output_path);
		void rectifyImages(const cv::Mat &H1, const cv::Mat &H2);
		void sampleFromX1X2(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2,
				const std::vector<int> &sample_indices, Eigen::MatrixXf &sample_X1,
				Eigen::MatrixXf &sample_X2);
		void estimateFundamentalMatrix(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2,
				Eigen::MatrixXf &F);
		float calcAvgFundamentalMatrixError(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2, const Eigen::MatrixXf &F);
		void estimateRansacFundamentalMatrix(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2,
				const float &dist_threshold, const float &ratio_threshold, Eigen::MatrixXf &F,
				std::vector<int> &inlier_indices);
		void denseSift(std::vector<cv::Point2f> &l_points, std::vector<cv::Point2f> &r_points);
		void recitfiedDenseSift(const cv::Mat &H, std::vector<cv::Point2f> &l_points, std::vector<cv::Point2f> &r_points);
		std::vector<float> getWindow(const cv::Mat &img, const size_t &i, const size_t &j,
				const size_t &window_size);
		float sumSquaredDifference(const std::vector<float> &window1, const std::vector<float> &window2);
		float correlation(const std::vector<float> &window1, const std::vector<float> &window2);
		float normalizedCorrelation(const std::vector<float> &window1, const std::vector<float> &window2);
		void intensityWindowCorrelation(const size_t &window_size, std::vector<cv::Point2f> &l_points,
				std::vector<cv::Point2f> &r_points);
		void greedyMatching(const Eigen::MatrixXf &F, const int &window_size,
				std::vector<cv::Point2f> &l_points, std::vector<cv::Point2f> &r_points);

	private:
		std::vector<cv::Mat> stereo_pair;
		std::vector<cv::Mat> rectified_stereo_pair;
		std::string output_path;
};
