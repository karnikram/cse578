/*
 * Program to capture frames from webcam until Esc
 *
 * Usage: ./capture_images <path-to-save>
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		std::cout << "Invalid input!\nCorrect usage: ./capture_images <path-to-store-images>\n";
		return -1;
	}

	const std::string images_path = argv[1];

	cv::VideoCapture cap(0);
	cv::Mat frame;
	size_t frame_no = 0;

	cv::namedWindow("Feed", cv::WINDOW_AUTOSIZE);
	std::cout << "Beginning to capture frames from webcam in 3 seconds..\n";
	std::this_thread::sleep_for(std::chrono::milliseconds(3000));

	while(1)
	{
		cap.read(frame);
		frame_no++;

		cv::imshow("Feed", frame);
		char c = cv::waitKey(10);
		if(c == 27)
			break;

		std::stringstream image_path;
		image_path << images_path << "/img" << std::setw(5) << std::setfill('0') << frame_no << ".png";
		std::cout << "Writing frame into disk at " << image_path.str() << std::endl;
		cv::imwrite(image_path.str(), frame);
	}

	std::cout << "Exiting..\n";

	return 0;
}
