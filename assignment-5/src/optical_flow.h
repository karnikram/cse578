#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

class OpticalFlow
{
	public:
		OpticalFlow(const cv::Mat &img1, const cv::Mat &img2,
				const std::string &output_path);

	private:
		cv::Mat img1, img2;
		std::string output_path;
};
