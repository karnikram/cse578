/* Program to composite two video sequences together.
 * Usage: ./chroma_keying <video1-path> <video2-path>
 *
 * Video1 is assumed to be the foreground sequence with
 * a constant green background. Video2 is the target background sequence.
 */

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

void create_composite(const cv::Mat &fg, const cv::Mat &bg, const cv::Mat &result)
{
	cv::Mat mask1(fg.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat mask2(fg.size(), CV_8UC1, cv::Scalar(0));

	for(size_t i = 0; i < fg.rows; i++)
		for(size_t j = 0; j < fg.cols; j++)
		{
			cv::Vec3b intensity = fg.at<cv::Vec3b>(i,j);
			if(intensity[0] <= 100 && intensity[1] >= 100 && intensity[2] <=100)
				mask2.at<uchar>(i,j) = 255;

			else
				mask1.at<uchar>(i,j) = 255;
		}

	cv::Mat temp1, temp2;
	fg.copyTo(temp1,mask1);
	bg.copyTo(temp2,mask2);

	cv::add(temp1,temp2,result);
}

int main(int argc, char *argv[])
{
	if(argc != 3)
	{
		std::cout << "Invalid input!\nCorrect usage: ./chroma_keying <video-1-path> <video-2-path>\n";
	}

	const std::string v1_path = argv[1], v2_path = argv[2];

	cv::VideoCapture cap1(v1_path), cap2(v2_path);
	if(!cap1.isOpened())
	{
		std::cout << "Could not open video 1!\n";
		return -1;
	}

	if(!cap2.isOpened())
	{
		std::cout << "Could not open video 2!\n";
		return -1;
	}

	cv::Size v1_frame_size(cap1.get(cv::CAP_PROP_FRAME_WIDTH),
			cap1.get(cv::CAP_PROP_FRAME_HEIGHT));
	cv::Size v2_frame_size(cap2.get(cv::CAP_PROP_FRAME_WIDTH),
			cap2.get(cv::CAP_PROP_FRAME_HEIGHT));

	if(!(v1_frame_size.height == v2_frame_size.height && v1_frame_size.width == v1_frame_size.width))
	{
		std::cout << "Video sequences are not of same size!\n";
		return -1;
	}

	cv::VideoWriter video_writer("composite.mp4", cv::VideoWriter::fourcc('X','2','6','4'), 30, v1_frame_size,true);

	cv::Mat fg, bg;
	cv::Mat result_frame(v1_frame_size, CV_8UC3);

	while(cap1.read(fg) && cap2.read(bg))
	{
		create_composite(fg, bg, result_frame);
		video_writer.write(result_frame);
	}

	std::cout << "Done!\n";

	return 0;
}
