#include "grabcut.h"
#include <opencv2/highgui.hpp>

GrabCut::GrabCut()
{
}

void GrabCut::setInputImage(const cv::Mat &input_img)
{
	this->input_img = input_img;
}

void GrabCut::showImage()
{
	cv::Mat temp_img;
	input_img.copyTo(temp_img);

	if(roi_state == IN_PROCESS || roi_state == SET)
		cv::rectangle(temp_img,cv::Point(roi.x,roi.y),cv::Point(roi.x + roi.width,
					roi.y + roi.height),cv::Scalar(0,0,255),2);

	cv::imshow("Input image",temp_img);
}

void GrabCut::captureMouseClick(int event, int x, int y, int flags, void *userdata)
{
	switch(event)
	{
		case CV_EVENT_LBUTTONDOWN:
			{
				if(roi_state == NOT_SET)
				{
					roi_state = IN_PROCESS;
					roi = cv::Rect(x,y,1,1);
				}
			}
			break;

		case CV_EVENT_LBUTTONUP:
			{
				if(roi_state == IN_PROCESS)
				{
					roi = cv::Rect(cv::Point(roi.x,roi.y),cv::Point(x,y));
					roi_state = SET;
					showImage();
				}
			}
			break;

		case CV_EVENT_MOUSEMOVE:
			{
				if(roi_state == IN_PROCESS)
				{
					roi = cv::Rect(cv::Point(roi.x,roi.y),cv::Point(x,y));
					showImage();
				}
			}
			break;
	}
}
