#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "Calibration.hpp"

int main(int argc, char *argv[])
{
	std::string frame_path = "../data/IMG_5455.JPG";
	std::string measurements_path = "../data/measurements.txt";

	cv::Mat frame;
	frame = cv::imread(frame_path, 1);

	cv::namedWindow("frame");
	cv::imshow("frame",frame);

	cv::waitKey(0);

	return 0;
}
