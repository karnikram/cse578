#include "optical_flow.h"
#include <string>
#include <opencv2/videoio.hpp>

void drawMotionSeg(cv::Mat &u, cv::Mat &v)
{
    cv::Mat mag, angle;
    cv::cartToPolar(u, v, mag, angle);
    
    for(int i = 0; i < u.rows; i++)
        for(int j = 0; j < u.cols; j++)
        {
            if(u.at<float>(i, j) < 50)
                u.at<float>(i, j) = 0;
            
            else if(v.at<float>(i, j) < 50)
                v.at<float>(i, j) = 0;
        }
    
    cv::Mat hsv = cv::Mat::zeros(u.size(), CV_8UC3);
    
    for(int i = 0; i < u.rows; i++)
        for(int j = 0; j < v.cols; j++)
        {
            cv::Vec3b value;
            value[0] = 2 * angle.at<float>(i, j) * 180 / 3.14;
            value[1] = 255;
            value[2] = mag.at<float>(i, j) * 255; 
            hsv.at<cv::Vec3b>(i, j) = value;
        }
    
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, CV_HSV2BGR);
    cv::namedWindow("motion", CV_WINDOW_NORMAL);
    cv::imshow("motion", bgr);
    cv::waitKey(0);
}

int main(int argc, char **argv)
{
	if(argc < 2)
	{
		std::cout << "Incorrect input format!\nCorrect usage: ./run_motion_seg <video_path>\n";
		return -1;
	}

	std::string input_path = argv[1];

    cv::VideoCapture cap(input_path);
    if(!cap.isOpened())
    {
        std::cout << "Video could not be opened!\n";
        return -1;
    }

	cv::Mat img1, img2;
	cv::Mat u, v;
    cap.read(img1);
    cv::imwrite("../data/frame.png",img1);
    cv::cvtColor(img1, img1, CV_BGR2GRAY);
    img1.convertTo(img1, CV_32FC1, 1.0/255, 0);
            
    while(cap.read(img2))
    {
        cv::cvtColor(img2, img2, CV_BGR2GRAY);
	    img2.convertTo(img2, CV_32FC1, 1.0/255, 0);

	    OpticalFlow optical_flow(img1, img2, 31);
	    optical_flow.computeOpticalFlow(u, v);
	    drawMotionSeg(u, v);
	    
	    img2.copyTo(img1);
    }

	return 0;
}
