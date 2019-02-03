#include <iostream>
#include <string>

#include "utils.hpp"
#include "calibrator.hpp"
#include <opencv2/core/eigen.hpp>

int main(int argc, char *argv[])
{
	// std::string frame_path = "../data/dlt/IMG_5455.JPG";
	// std::string points_2d_path = "../data/dlt/2d-points.txt";
	// std::string points_3d_path = "../data/dlt/3d-points.txt";
    // std::string points_2d_path = "../data/dlt/undistort-2d-points.txt";

	std::string frame_path = "../data/phone-camera/dlt/cube.jpg";
	std::string points_2d_path = "../data/phone-camera/dlt/2d-points.txt";
	std::string points_3d_path = "../data/phone-camera/dlt/3d-points.txt";

	Eigen::MatrixXf points_2d;
	utils::loadMatrix(points_2d_path,utils::getNumberofLines(points_2d_path),2,points_2d);

	Eigen::MatrixXf points_3d;
	utils::loadMatrix(points_3d_path,utils::getNumberofLines(points_3d_path),3,points_3d);

	if(points_2d.rows() != points_3d.rows())
	{
		std::cout << "Number of correspondences don't match!\n";
		return -1;
	}

	cv::Mat frame;
	frame = cv::imread(frame_path,1);

/* Undistortion

	cv::Mat undistorted_frame;
	cv::Mat distortion = (cv::Mat_<double>(1,5) << 0.2963, -2.999, 0.0017, 0.0112, 20.5691); //obtained from matlab (Zhang's)
	cv::Mat cv_K = (cv::Mat_<double>(3,3) << 13459, 78, 2983, 0, 13507, 1849, 0, 0, 1); //obtained from matlab (Zhang's)
	cv::undistort(frame,undistorted_frame,cv_K,distortion);
 */

    //DLT with RANSAC calibration
	Calibrator calib(points_3d, points_2d);
    std::vector<int> inlier_indices;
    calib.calibrateByDltRansac(40, 0.5, inlier_indices);
	calib.drawOverlay(inlier_indices,frame);

	// Decompose P
	Eigen::MatrixXf K, R, c;
	calib.decomposePMatrix(K,R,c);

	cv::namedWindow("frame",CV_WINDOW_NORMAL);
	cv::imshow("frame",frame);
    cv::imwrite("../report/imgs/cube-ransac.png",frame);
	cv::waitKey(0);	

	return 0;
}
