#include "optical_flow.h"
#include <vector>
#include <numeric>

OpticalFlow::OpticalFlow(const cv::Mat &img1, const cv::Mat &img2,
		const std::string &output_path)
{
	this->img1 = img1;
	this->img2 = img2;
	
	this->img1_x.create(img1.size(), CV_8UC1);
	this->img1_y.create(img1.size(), CV_8UC1);
	this->img2_x.create(img2.size(), CV_8UC1);
	this->img2_y.create(img2.size(), CV_8UC1);
	
	this->output_path = output_path;
}

std::vector<uchar> OpticalFlow::getWindow(const cv::Mat &img, const int &i,
        const int &j)
{
	std::vector<uchar> v_window;
	
	for(int ii = i - 1; ii <= i + 1; ii++)
		for(int jj = j - 1; jj <= j + 1; jj++)
		{
			v_window.push_back(img.at<uchar>(ii,jj));
		}

	return v_window;
}

void OpticalFlow::computeImageGradients()
{
    std::vector<int> sobel_x {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    std::vector<int> sobel_y {1, 2, 1, 0, 0, 0, -1, -2, -1};
    
	for(int i = 1; i < img1.rows - 1; i++)
		for(int j = 1; j < img1.cols - 1; j++)
		{
		    std::vector<uchar> window = getWindow(img1, i, j);
			img1_x.at<uchar>(i, j) = abs(std::inner_product(std::begin(window), std::end(window), std::begin(sobel_x), 0));
			img1_y.at<uchar>(i, j) = abs(std::inner_product(std::begin(window), std::end(window), std::begin(sobel_y), 0));
		}

	for(int i = 1; i < img2.rows - 1; i++)
		for(int j = 1; j < img2.cols - 1; j++)
		{
		    std::vector<uchar> window = getWindow(img2, i, j);
		    img2_x.at<uchar>(i, j) = std::inner_product(std::begin(window), std::end(window), std::begin(sobel_x), 0);
			img2_y.at<uchar>(i, j) = std::inner_product(std::begin(window), std::end(window), std::begin(sobel_y), 0);
		}
		
		
	cv::namedWindow("img1_x", CV_WINDOW_AUTOSIZE);
	cv::imshow("img1_x", img1_x);
	cv::namedWindow("img1_y", CV_WINDOW_AUTOSIZE);
	cv::imshow("img1_y", img1_y);
	
	std::cout << img1_x.size() << std::endl;
	cv::waitKey();
}
