#include "optical_flow.h"
#include <vector>
#include <numeric>

OpticalFlow::OpticalFlow(const cv::Mat &img1, const cv::Mat &img2,
        const int &window_size)
{
	this->img1 = img1;
	this->img2 = img2;
	this->window_size = window_size;
}

int OpticalFlow::isValidPoint(const int &i, const int &j)
{
    if(i >= 0 && i < img1.rows && j >= 0 && j < img1.cols)
        return 1;
     
    else
        return 0;
}

std::vector<float> OpticalFlow::getWindow(const cv::Mat &mat, const int &i, const int &j)
{
	std::vector<float> window;
	for(size_t ii = i - window_size/2; ii <= i + window_size/2; ii++)
		for(size_t jj = j - window_size/2; jj <= j + window_size/2; jj++)
		{
			window.push_back(mat.at<float>(ii, jj));
		}

	return window;
}

cv::Mat OpticalFlow::sumOverWindow(const cv::Mat &mat)
{
    cv::Mat sum = cv::Mat::zeros(mat.size(), CV_32FC1);
    
    for(int i = window_size/2; i < mat.rows - window_size/2; i++)
        for(int j = window_size/2; j < mat.cols - window_size/2; j++)
        {
            std::vector<float> window = getWindow(mat, i, j);
            sum.at<float>(i, j) = std::accumulate(window.begin(), window.end(), 0.0);
        }
        
    return sum;        
}

void OpticalFlow::computeImageGradients()
{
    // img_x
    cv::Mat kernel = cv::Mat::ones(2, 2, CV_32FC1);
    kernel.at<float>(0, 0) = -1.0;
    kernel.at<float>(1, 0) = -1.0;

    cv::Mat dst1, dst2;
    filter2D(img1, dst1, -1, kernel);
    filter2D(img2, dst2, -1, kernel);

    img_x = dst1 + dst2;

    // img_y
    kernel = cv::Mat::ones(2, 2, CV_32FC1);
    kernel.at<float>(0, 0) = -1.0;
    kernel.at<float>(0, 1) = -1.0;

    filter2D(img1, dst1, -1, kernel);
    filter2D(img2, dst2, -1, kernel);

    img_y = dst1 + dst2;

	// img_t
	kernel = cv::Mat::ones(2, 2, CV_32FC1);
    kernel = kernel.mul(-1);

    filter2D(img1, dst1, -1, kernel);
    kernel = kernel.mul(-1);
    filter2D(img2, dst2, -1, kernel);

    img_t = dst1 + dst2;
}

void OpticalFlow::computeOpticalFlow(cv::Mat &u, cv::Mat &v)
{
    computeImageGradients();
    
    cv::Mat img_xx = img_x.mul(img_x);
    cv::Mat img_yy = img_y.mul(img_y);
    cv::Mat img_xy = img_x.mul(img_y);
    cv::Mat img_xt = img_x.mul(img_t);
    cv::Mat img_yt = img_y.mul(img_t);
    
    cv::Mat sum_img_xx = sumOverWindow(img_xx);
    cv::Mat sum_img_yy = sumOverWindow(img_yy);
    cv::Mat sum_img_xy = sumOverWindow(img_xy);
    cv::Mat sum_img_xt = sumOverWindow(img_xt);
    cv::Mat sum_img_yt = sumOverWindow(img_yt);
    
    cv::Mat tmp = sum_img_xx.mul(sum_img_yy) - sum_img_xy.mul(sum_img_xy);
    u = sum_img_xy.mul(sum_img_yt) - sum_img_yy.mul(sum_img_xt);
    v = sum_img_xt.mul(sum_img_xy) - sum_img_xx.mul(sum_img_yt);
    
    cv::divide(u, tmp, u);
    cv::divide(v, tmp, v);
}
