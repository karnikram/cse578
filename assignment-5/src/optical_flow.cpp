#include "optical_flow.h"

OpticalFlow::OpticalFlow(const cv::Mat &img1, const cv::Mat &img,
		const std::string &output_path)
{
	this->img1 = img1;
	this->img2 = img2;
	this->output_path = output_path;
}

