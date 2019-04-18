#include <iostream>
#include <string>

#include "grabcut.h"

GrabCut grabcut;

void onMouseClick(int event, int x, int y, int flags, void *userdata)
{
	grabcut.captureMouseClick(event, x, y, flags, userdata);
}

int main(int argc, char **argv)
{
	if(argc < 3)
	{
		std::cout << "Incorrect usage!\n Correct usage: ./run <path-to-image> <path-to-save-foreground>\n";
		return -1;
	}

	std::string input_img_path = argv[1];
	std::string output_img_path = argv[2];

	cv::Mat input_img = cv::imread(argv[1], 1);
	grabcut.setInputImage(input_img);

	//cv::Mat input_img_hsv;
	//cv::cvtColor(input_img, input_img_hsv, cv::COLOR_BGR2YCrCb);
	//grabcut.setInputImage(input_img_hsv);
	grabcut.setOutputPath(output_img_path);

	cv::namedWindow("Input image",CV_WINDOW_NORMAL);
	cv::imshow("Input image",input_img);
	cv::setMouseCallback("Input image",onMouseClick,0);

	while(1)
	{
		char c = cv::waitKey(0);
		switch(c)
		{
			case '\x1b':
				{
					goto exit;
				}
				break;
			
			case 's':
				{
					if(grabcut.getRoiState() == GrabCut::SET)
					{
						std::cout << "Starting grabcut!\n" << std::endl;
						grabcut.runGrabCut();
						goto exit;
					}
					else
						std::cout << "First select ROI!\n" << std::endl;
				}
				break;
		}
	}

exit:
	return 0;
}
