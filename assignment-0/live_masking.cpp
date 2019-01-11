/* Code to dynamically vary the range of values
 * to mask out from the foreground image,
 * using track bars.
 * Usage: ./image_thresholding <path-to-image>
 */

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

void extract_fg(cv::Mat frame, const cv::Vec3b &tl_intensity, const cv::Vec3b &tu_intensity, const cv::Mat &fg)
{
	cv::Mat mask(frame.size(), CV_8UC1, cv::Scalar(0));

	for(size_t i = 0; i < frame.rows; i++)
		for(size_t j = 0; j < frame.cols; j++)
		{
			cv::Vec3b intensity = frame.at<cv::Vec3b>(i,j);
			if(intensity[0] >= tl_intensity[0] && intensity[0] <= tu_intensity[0] && intensity[1] >= tl_intensity[1] && intensity[1] <= tu_intensity[1] && intensity[2] >= tl_intensity[2] && intensity[2] <= tu_intensity[2])
				mask.at<uchar>(i,j) = 0;

			else
				mask.at<uchar>(i,j) = 255;
		}

	frame.copyTo(fg,mask);
}

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		std::cout << "Invalid input!\n Correct usage: ./image_thresholding <path-to-image>\n";
		return -1;
	}

	const std::string image_path = argv[1];

	cv::namedWindow("Control");

	int lr=0,lg=100,lb=0,ur=100,ug=255,ub=100;

	cv::Vec3b tl_intensity;
	cv::Vec3b tu_intensity;

	cv::createTrackbar("lR","Control",&lr,255);
	cv::createTrackbar("uR","Control",&ur,255);
	cv::createTrackbar("lG","Control",&lg,255);
	cv::createTrackbar("uG","Control",&ug,255);
	cv::createTrackbar("lB","Control",&lb,255);
	cv::createTrackbar("uB","Control",&ub,255);

	while(1)
	{
		tl_intensity[0] = lr;
		tl_intensity[1] = lg;
		tl_intensity[2] = lb;

		tu_intensity[0] = ur;
		tu_intensity[1] = ug;
		tu_intensity[2] = ub;

		cv::Mat frame, s_frame;//, frame_hsv;
		frame = cv::imread(image_path,1);

		cv::Mat fg(frame.size(), CV_8UC3);
		//cv::cvtColor(frame,frame_hsv,cv::COLOR_BGR2HSV);
		//extract_fg(frame_hsv,tl_intensity,tu_intensity,fg);
		
		//for(int i = 1; i < 5; i = i + 2)
		//	cv::blur(frame,s_frame,cv::Size(i,i), cv::Point(-1,-1));

		extract_fg(s_frame,tl_intensity,tu_intensity,fg);

		cv::namedWindow("fg",cv::WINDOW_AUTOSIZE);
		cv::imshow("fg",fg);

		char c = cv::waitKey(30);
		if(c == 27)
			break;
	}

	return 0;
}
