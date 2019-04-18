#include "optical_flow.h"
#include <string>
#include <chrono>

void drawFlow(const cv::Mat &img1, const cv::Mat &u, const cv::Mat &v)
{
    cv::Mat res(img1.size(), CV_8UC3);
    cv::cvtColor(img1, res, CV_GRAY2RGB);
    
    cv::Mat flow(img1.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    
    for(int i = 10; i < img1.rows; i+=10)
        for(int j = 10; j < img1.cols; j+=10)
        {
            cv::circle(res, cv::Point2f(j, i), 1, cv::Scalar(255, 0, 0), 1);
            cv::circle(flow, cv::Point2f(j, i), 1, cv::Scalar(255, 0, 0), 1);
            cv::line(res, cv::Point2f(j, i), cv::Point2f(cvRound(j + u.at<float>(i, j)),
                cvRound(i + v.at<float>(i, j))), cv::Scalar(0, 0, 255), 1);
            cv::line(flow, cv::Point2f(j, i), cv::Point2f(cvRound(j + u.at<float>(i, j)),
                cvRound(i + v.at<float>(i, j))), cv::Scalar(0, 0, 255), 1);
        }
        
    cv::namedWindow("Result", CV_WINDOW_AUTOSIZE);
    cv::imshow("Result", res);
    cv::namedWindow("Flow", CV_WINDOW_AUTOSIZE);
    cv::imshow("Flow", flow);
    cv::waitKey();
}

int main(int argc, char **argv)
{
	if(argc < 3)
	{
		std::cout << "Incorrect input format!\nCorrect usage: ./run_of <img1_path> <img2_path>\n";
		return -1;
	}

	std::string img1_path = argv[1];
	std::string img2_path = argv[2];

	cv::Mat img1 = cv::imread(img1_path, 0);
	cv::Mat img2 = cv::imread(img2_path, 0);
	
	img1.convertTo(img1, CV_32FC1, 1.0/255, 0);
	img2.convertTo(img2, CV_32FC1, 1.0/255, 0);

    cv::Mat u, v;
    auto t1 = std::chrono::high_resolution_clock::now();
	OpticalFlow optical_flow(img1, img2, 3);
	optical_flow.computeOpticalFlow(u, v);
    auto t2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
	std::cout << "LK took "<< duration << " ms\n";
	
	drawFlow(img1, u, v);

	return 0;
}
