/*
 * Target - green screen video
 * Source - background video
 */

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

void create_composite(const cv::Mat &fg, const cv::Mat &bg, const cv::Mat &result)
{
	cv::Mat mask1(fg.rows, fg.cols, CV_8UC1, cv::Scalar(0));
	cv::Mat mask2(fg.rows, fg.cols, CV_8UC1, cv::Scalar(0));

	for(size_t i = 0; i < fg.rows; i++)
		for(size_t j = 0; j < fg.cols; j++)
		{
			cv::Vec3b intensity = fg.at<cv::Vec3b>(i,j);
			if(intensity[1] >= 200)
				mask2.at<uchar>(i,j) = 255;

			else
				mask1.at<uchar>(i,j) = 255;
		}

	cv::Mat temp1, temp2;
	fg.copyTo(temp1,mask1);
	bg.copyTo(temp2,mask2);

	cv::namedWindow("masked fg",cv::WINDOW_AUTOSIZE);
	cv::namedWindow("masked bg",cv::WINDOW_AUTOSIZE);

	cv::imshow("masked fg",temp1);
	cv::imshow("masked bg",temp2);

	cv::add(temp1,temp2,result);
	
	cv::namedWindow("result",cv::WINDOW_AUTOSIZE);
	cv::imshow("result",result);
	
	std::cout << temp1.type() << " " << temp2.type() << std::endl;

	cv::waitKey();
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

	cv::Mat frame1, frame2;

	cap1.read(frame1);
	cap2.read(frame2);

	cv::Mat result_frame(frame1.size(), CV_8UC3);
	
	create_composite(frame1, frame2, result_frame);

	//cv::VideoWriter video_writer();
	//while(cap1.read(frame1) && cap2.read(frame2))
	//{

		//video_writer.write(result_frame);
	//}

	return 0;
}
