#include "optical_flow.h"
#include <string>

int main(int argc, char **argv)
{
	if(argc < 4)
	{
		std::cout << "Incorrect input format!\nCorrect usage: ./run <img1_path> <img2_path> <output_path>\n";
		return -1;
	}

	std::string img1_path = argv[1];
	std::string img2_path = argv[2];
	std::string output_path = argv[3];

	cv::Mat img1 = cv::imread(img1_path, 0);
	cv::Mat img2 = cv::imread(img2_path, 0);

	OpticalFlow optical_flow(img1, img2, output_path);
	optical_flow.computeImageGradients();

	return 0;
}
