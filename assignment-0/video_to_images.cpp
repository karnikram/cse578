/*
 * Code to convert a given video sequence into
 * its constituent frames.
 *
 * Usage : ./video_to_images <path-to-video> <path-to-store-images>
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[])
{
	if(argc != 3)
	{
		std::cout << "Invalid input format!\nCorrect usage: ./video_to_images <path-to-video> <path-to-store-images>\n";
		return -1;
	}

	const std::string video_path = argv[1], images_path = argv[2];

	cv::VideoCapture cap(video_path);
	if(!cap.isOpened())
	{
		std::cout << "Video could not be opened!\n";
		return -1;
	}

	std::cout << "Video opened!\n";
	std::cout << "Number of frames in the video frame: " << cap.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT) << std::endl;

	cv::Mat frame;
	size_t frame_no = 0;

	while(cap.read(frame))
	{
		std::stringstream image_path;
		image_path << images_path << "/img" << std::setw(5) << std::setfill('0') << frame_no++ << ".png";
		std::cout << "Writing frame #" << frame_no << " into disk at " << image_path.str() << std::endl;
		cv::imwrite(image_path.str(), frame);
	}

	std::cout << frame_no << " frame(s) written to the disk!\n";
	return 0;
}
