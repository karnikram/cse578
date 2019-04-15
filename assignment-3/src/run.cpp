#include "stereomatcher.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <chrono>

int main(int argc, char *argv[])
{
	if(argc < 3)
	{
		std::cout << "Invalid input!\n";
		std::cout << "Correct usage: ./run <path-to-stereo-img> <path-to-save-result>\n";
		return -1;
	}

	std::vector<cv::Mat> stereo_pair;
	cv::Mat img = cv::imread(argv[1],1);
	std::string output_path = argv[2];

	stereo_pair.push_back(img(cv::Rect(0,0,img.cols/2,img.rows)));
	stereo_pair.push_back(img(cv::Rect(img.cols/2,0,img.cols/2,img.rows)));

	std::cout << stereo_pair.size() << " images loaded!\n";
	
	std::vector<cv::Point2f> l_points, r_points;
	StereoMatcher matcher(stereo_pair,output_path);

	//auto t1 = std::chrono::high_resolution_clock::now();
	matcher.denseSift(l_points,r_points);
	//auto t2 = std::chrono::high_resolution_clock::now();

	//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
	//std::cout << "denseSift() took "<< duration << " ms\n";
	
	//Intensity window based correlation
	//auto t1 = std::chrono::high_resolution_clock::now();
	//matcher.intensityWindowCorrelation(5,l_points,r_points);
	//auto t2 = std::chrono::high_resolution_clock::now();
	
	//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
	//std::cout << "intensityWindowCorrelation() took "<< duration << " ms\n";

	//Fundamental matrix estimation
	cv::Mat cv_F;
	Eigen::MatrixXf F;
	//std::vector<int> inlier_indices;
	//matcher.estimateRansacFundamentalMatrix(X1,X2,0.1,0.1,F,inlier_indices);
	cv_F = findFundamentalMat(l_points,r_points,CV_FM_RANSAC);
	cv2eigen(cv_F,F);

	//Rectification
	cv::Mat H1,H2;
	stereoRectifyUncalibrated(l_points,r_points,cv_F,stereo_pair[0].size(),H1,H2);
	matcher.rectifyImages(H1,H2);

	l_points.clear();
	r_points.clear();

	auto t1 = std::chrono::high_resolution_clock::now();
	matcher.greedyMatching(F,5,l_points,r_points);
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t1).count();

	std::cout << "greedyMatching() took " << duration << " ms\n";
	
	//TODO
	//DTW
	
	return 0;
}
