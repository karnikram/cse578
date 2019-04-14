#include "optical_flow.h"
#include <string>
#include <opencv2/videoio.hpp>

enum{NOT_SET = 0, IN_PROCESS = 1, SET = 2};
uchar roi_state = NOT_SET;
cv::Rect roi;
cv::Mat img1;

void showInputImage(const cv::Mat &img)
{
	cv::Mat temp_img;
	img.copyTo(temp_img);

	if(roi_state == IN_PROCESS || roi_state == SET)
		cv::rectangle(temp_img,cv::Point(roi.x,roi.y),cv::Point(roi.x + roi.width,
					roi.y + roi.height),cv::Scalar(0,0,255),2);

	cv::imshow("First frame",temp_img);
}

void onMouseClick(int event, int x, int y, int flags, void *userdata)
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
					showInputImage(img1);
					std::cout << "Region set!\nPress 's' to start tracking!\n";
				}
			}
			break;

		case CV_EVENT_MOUSEMOVE:
			{
				if(roi_state == IN_PROCESS)
				{
					roi = cv::Rect(cv::Point(roi.x,roi.y),cv::Point(x,y));
					showInputImage(img1);
				}
			}
			break;
	}
}

void runTracking(cv::VideoCapture &cap)
{
    cv::Mat prev, prev_gr, next, next_gr, u, v;
    
    img1.copyTo(prev);
    cv::cvtColor(prev, prev_gr, CV_BGR2GRAY);
    prev_gr.convertTo(prev_gr, CV_32FC1, 1.0/255, 0);
            
    while(cap.read(next))
    {
        cv::cvtColor(next, next_gr, CV_BGR2GRAY);
	    next_gr.convertTo(next_gr, CV_32FC1, 1.0/255, 0);

	    OpticalFlow optical_flow(prev_gr, next_gr, 5);
	    optical_flow.computeOpticalFlow(u, v);
        
        cv::Scalar mean_u = cv::mean(u(roi));
        cv::Scalar mean_v = cv::mean(v(roi));
        
        roi.x += mean_u[0];
        roi.y += mean_v[0];
	    
		cv::rectangle(next,cv::Point(roi.x,roi.y),cv::Point(roi.x + roi.width,
					roi.y + roi.height),cv::Scalar(0,0,255),2);

        cv::namedWindow("Track", CV_WINDOW_AUTOSIZE);
	    cv::imshow("Track", next);
	    cv::waitKey(0);
	
	    next_gr.copyTo(prev_gr);
    }
}

int main(int argc, char **argv)
{
	if(argc < 2)
	{
		std::cout << "Incorrect input format!\nCorrect usage: ./run <video_path>\n";
		return -1;
	}

	std::string input_path = argv[1];

    cv::VideoCapture cap(input_path);
    if(!cap.isOpened())
    {
        std::cout << "Video could not be opened!\n";
        return -1;
    }

	cap.read(img1);
    std::cout << "Select ROI!\n";
    cv::namedWindow("First frame", CV_WINDOW_AUTOSIZE);
    cv::imshow("First frame", img1);
    cv::setMouseCallback("First frame", onMouseClick, 0);
            
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
					if(roi_state == SET)
					{
						std::cout << "Starting tracking!\n" << std::endl;
						runTracking(cap);
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
