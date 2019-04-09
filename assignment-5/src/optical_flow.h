#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>

class OpticalFlow
{
	public:
		OpticalFlow(const cv::Mat &img1, const cv::Mat &img2,
				const std::string &output_path);

        std::vector<uchar> getWindow(const cv::Mat &img, const int &i,
                const int &j);
                
        void computeImageGradients();

	private:
		cv::Mat img1, img2;
		cv::Mat img1_x, img1_y, img2_x, img2_y;
		std::string output_path;
};
