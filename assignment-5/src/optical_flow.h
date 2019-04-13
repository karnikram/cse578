#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>

class OpticalFlow
{
	public:
		OpticalFlow(const cv::Mat &img1, const cv::Mat &img2,
		        const int &window_size);
                
        int isValidPoint(const int &i, const int &j);

        void computeImageGradients();
        
        std::vector<float> getWindow(const cv::Mat &mat, const int &i, const int &j);
        
        cv::Mat sumOverWindow(const cv::Mat &mat);

		void computeOpticalFlow(cv::Mat &u, cv::Mat &v);

	private:
		cv::Mat img1, img2;
		int window_size;
		cv::Mat img_x, img_y, img_t;
};
