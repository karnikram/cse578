#include <iostream>
#include <string>

#include "utils.hpp"
#include "calibrator.hpp"

int main(int argc, char *argv[])
{
	std::string frame_path = "../data/dlt/IMG_5455.JPG";
	std::string points_2d_path = "../data/2d-points.txt";
	std::string points_3d_path = "../data/3d-points.txt";

	Eigen::MatrixXf points_2d;
	utils::loadMatrix(points_2d_path,utils::getNumberofLines(points_2d_path),2,points_2d);

	Eigen::MatrixXf points_3d;
	utils::loadMatrix(points_3d_path,utils::getNumberofLines(points_3d_path),3,points_3d);

	if(points_2d.rows() != points_3d.rows())
	{
		std::cout << "Number of correspondences don't match!\n";
		return -1;
	}

	Calibrator calib(points_3d, points_2d);

	std::vector <int> sample_indices(points_3d.rows());
	std::iota(std::begin(sample_indices), std::end(sample_indices), 0);
	//calib.calibrateByDlt(sample_indices);
	calib.calibrateByDltRansac(50);

	Eigen::MatrixXf K, R, c;
	calib.decomposePMatrix(K,R,c);

	cv::Mat frame,undistorted_frame;
	frame = cv::imread(frame_path,1);
	calib.drawOverlay(frame);

	cv::namedWindow("frame",CV_WINDOW_NORMAL);
	cv::imshow("frame",frame);
	cv::waitKey(0);	

/*  
	cv::Mat frame,undistorted_frame;
	frame = cv::imread(frame_path,1);
	cv::namedWindow("distorted frame",CV_WINDOW_NORMAL);
	cv::imshow("distorted frame",frame);

	cv::waitKey();
*/

/* 	calib.calibrateByDltRansac(5.0);

	Eigen::MatrixXf K,R,c;
	calib.decomposePMatrix(K,R,c);

 	cv::Mat frame,undistorted_frame;
	frame = cv::imread(frame_path,1);
	
	cv::Mat K = (cv::Mat_<double>(3,3) << 1.3459e+04, 77.5444, 2.9829e+03, 0, 1.3507e+04, 1.8488e+03, 0, 0, 1);

	cv::Mat distortion = (cv::Mat_<double>(1,5) << 0.2963, -2.3011, 0.0017, 0.0112, 20.6084);

	cv::undistort(frame,undistorted_frame,K,distortion); //extra argument - newcameramatrix?

	cv::namedWindow("distorted frame",CV_WINDOW_NORMAL);
	cv::imshow("distorted frame",frame);

	cv::namedWindow("undistorted frame",CV_WINDOW_NORMAL);
	cv::imshow("undistorted frame",undistorted_frame);

	cv::waitKey(0);

	drawOverlay(P,X,frame);
 */
	return 0;
}
