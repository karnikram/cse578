/*
 * Code to concatenate a set of images
 * into a video sequence. Assumes all images are of same size.
 *
 * Usage: ./images_to_video <path_to_image_folder> <frame_rate> <video_name>
 */

#include <iostream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <experimental/filesystem>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::experimental::filesystem;

int main(int argc, char *argv[])
{
	if(argc != 4)
	{
		std::cout << "Invalid input!\nCorrect usage: ./images_to_video <path-to-image-folder> <frame-rate> <video-name>\n";
		return -1;
	}

	const std::string images_path = argv[1];
	size_t frame_rate = atoi(argv[2]);
	const std::string video_name = argv[3];

	std::vector<std::string> file_names;
	for(const auto & image_name : fs::directory_iterator(images_path))
	{
		file_names.push_back(image_name.path());
	}

	std::sort(file_names.begin(), file_names.end());

	cv::Mat frame;
	frame = cv::imread(file_names[0], 1);
	cv::VideoWriter video_writer(video_name + ".mp4", cv::VideoWriter::fourcc('X','2','6','4'), frame_rate, frame.size(), true);

	if(!video_writer.isOpened())
	{
		std::cout << "Could not open video file for writing!\n";
		return -1;
	}

	for(auto file_name : file_names)
	{
		frame = cv::imread(file_name, 1);
		video_writer.write(frame);
	}

	std::cout << "Finished writing video file: " << video_name << ".mp4 into disk!" << std::endl;

	return 0;
}
