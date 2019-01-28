#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/LU>

#include "Calibration.hpp"

int getNumberofLines(const std::string &file_name)
{
	std::ifstream file(file_name);
	std::string line;
	int num_lines = 0;

	while(std::getline(file,line))
		num_lines++;

	file.close();

	return num_lines;
}

void loadMatrix(const std::string &file_name, const int &nrows, const int &ncols, Eigen::MatrixXf &mat)
{
	std::ifstream file(file_name);
	mat.resize(nrows, ncols);

	if(file.is_open())
		for(int i = 0; i < nrows; i++)
			for(int j = 0; j < ncols; j++)
			{
				float value = 0.0;
				file >> value;
				mat(i,j) = value;
			}
	else
		std::cout << "Could not open file!\n";

	file.close();
}

int main(int argc, char *argv[])
{
	std::string frame_path = "../data/dlt/IMG_5455.JPG";
	std::string points_2d_path = "../data/2d-points.txt";
	std::string points_3d_path = "../data/3d-points.txt";

	Eigen::MatrixXf points_2d;
	loadMatrix(points_2d_path,getNumberofLines(points_2d_path),2,points_2d);

	Eigen::MatrixXf points_3d;
	loadMatrix(points_3d_path,getNumberofLines(points_3d_path),3,points_3d);

	int num_points = 0;

	if(points_2d.rows() != points_3d.rows())
	{
		std::cout << "Number of correspondences don't match!\n";
		return -1;
	}

	else
		num_points = points_2d.rows();

	//Convert to homogeneous representation
	Eigen::MatrixXf X, x;
	x = points_2d;
	x.conservativeResize(x.rows(),x.cols()+1);
	x.col(x.cols()-1).setOnes();

	X = points_3d;
	X.conservativeResize(X.rows(),X.cols()+1);
	X.col(X.cols()-1).setOnes();
	X = X * 0.01;

	Eigen::MatrixXf p(12,1);
	Eigen::MatrixXf M(2*num_points,12);

	//Build M
	for(int i = 0,j=0; i < M.rows(); i+=2,j++)
	{
		M.row(i) << X.row(j) * -1, Eigen::MatrixXf::Zero(1,4), X.row(j) * x(j,0);
		M.row(i+1) << Eigen::MatrixXf::Zero(1,4), X.row(j) * -1, X.row(j) * x(j,1);
	}

	std::cout << M << std::endl;
	std::cout << "M size:\n" << M.rows() << " " << M.cols() << std::endl;

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(M,Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXf V;
	V = svd.matrixV();
	p = V.col(V.cols()-1);

	std::cout << "p:\n" << p << std::endl;

	Eigen::MatrixXf P(3,4);
	P = p;
	P.resize(3,4);
	std::cout << "P:\n" << P << std::endl;

	Eigen::MatrixXf H,p4;
	H = P.block(0,0,3,3);
	p4 = P.col(3);

	std::cout << "H:\n" << H << std::endl;
	std::cout << "p4:\n" << p4 << std::endl;
	
	//Camera center
	Eigen::MatrixXf c;
	c = -1 * H.inverse() * p4;

	//Calibration matrix, rotation matrix
	Eigen::MatrixXf K, R;
	Eigen::HouseholderQR<Eigen::MatrixXf> qr(H.inverse());	
	R = qr.householderQ();
	K = qr.matrixQR().triangularView<Eigen::Upper>();

	R = R.inverse();
	K = K.inverse();
	K = K / K(2,2);

	std::cout << "R:\n" << R << std::endl;
	std::cout << "K:\n" << K << std::endl;

	cv::Mat frame;
	frame = cv::imread(frame_path,1);

	//cv::namedWindow("frame",CV_WINDOW_NORMAL);
	//cv::imshow("frame",frame);

	//cv::waitKey(0);
	

	return 0;
}
