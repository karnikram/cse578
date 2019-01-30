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
	calib.calibrateByDlt();

	Eigen::MatrixXf K,R,c;
	calib.decomposePMatrix(K,R,c);

	//cv::Mat frame;
	//frame = cv::imread(frame_path,1);

	//cv::namedWindow("frame",CV_WINDOW_NORMAL);
	//cv::imshow("frame",frame);

	//cv::waitKey(0);
	
	//drawOverlay(P,X,frame);

	return 0;
}
