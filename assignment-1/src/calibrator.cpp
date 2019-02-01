#include "calibrator.hpp"
#include <stdlib.h>

Calibrator::Calibrator(const Eigen::MatrixXf &points_3d, const Eigen::MatrixXf &points_2d)
{
	x = points_2d;
	x.conservativeResize(x.rows(),x.cols()+1);
	x.col(x.cols()-1).setOnes();

	X = points_3d;
	X = X * 0.01; // assumes points are in cm
	X.conservativeResize(X.rows(),X.cols()+1);
	X.col(X.cols()-1).setOnes();
}

void Calibrator::calibrateByDlt(const std::vector<int> &sample_indices)
{

	Eigen::MatrixXf X_samples(sample_indices.size(),4);
	Eigen::MatrixXf x_samples(sample_indices.size(),3);

	if(sample_indices.size() == X.rows())
	{
		X_samples = X;
		x_samples = x;
	}

	else
	{	int j = 0;	
		for (int i : sample_indices)
		{
			X_samples.row(j) = X.row(i);
			x_samples.row(j++) = x.row(j);
		}
	}

	Eigen::ArrayXf p(12);
	Eigen::MatrixXf M(2*x_samples.rows(),12);

	// Build M
	for(int i = 0,j=0; i < M.rows(); i+=2,j++)
	{
		M.row(i) << X_samples.row(j) * -1, Eigen::MatrixXf::Zero(1,4), X_samples.row(j) * x_samples(j,0);
		M.row(i+1) << Eigen::MatrixXf::Zero(1,4), X_samples.row(j) * -1, X_samples.row(j) * x_samples(j,1);
	}

	std::cout << M << std::endl;
	std::cout << "M size:\n" << M.rows() << " " << M.cols() << std::endl;

	// Estimate p
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(M,Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXf V;
	V = svd.matrixV();
	p = V.col(V.cols()-1);
	std::cout << "p:\n" << p << std::endl;

	// Build P
 	P.resize(3,4);
	P.row(0) = p.segment(0,4);
	P.row(1) = p.segment(4,4);
	P.row(2) = p.segment(8,4);
	std::cout << "P:\n" << P << std::endl;
}

void Calibrator::calibrateByDltRansac(const float &dist_threshold)
{
	
	for(int n = 0; n < 500; n++)
	{
		std::vector<int> sample_indices = utils::generateRandomVector(0,X.rows(),6);
		calibrateByDlt(sample_indices);
	
		std::vector<int> inlier_indices;
		for(int i = 0; i < X.rows(); i++)
		{
			float dist = calcReprojectionError(X.row(i),x.row(i));
			if(dist < dist_threshold)
				inlier_indices.push_back(i); 
		}

		if(inlier_indices.size() > 0.8 * X.rows())
		{
			calibrateByDlt(inlier_indices);
			return;
		}
	
		else
			continue;
	}

}

float Calibrator::calcReprojectionError(const Eigen::Vector4f &X, const Eigen::Vector3f &x)
{
	Eigen::Vector3f est_x = P * X;
	return (est_x - x).squaredNorm();
}

void Calibrator::decomposePMatrix(Eigen::MatrixXf &K, Eigen::MatrixXf &R, Eigen::MatrixXf &c)
{
	Eigen::MatrixXf H,p4;
	H = P.block(0,0,3,3);
	p4 = P.col(3);

	std::cout << "H:\n" << H << std::endl;
	std::cout << "p4:\n" << p4 << std::endl;
	
	//Camera center
	c = -1 * H.inverse() * p4;

	//Calibration matrix, rotation matrix
	Eigen::HouseholderQR<Eigen::MatrixXf> qr(H.inverse());	
	R = qr.householderQ();
	K = qr.matrixQR().triangularView<Eigen::Upper>();

	R = R.inverse();
	K = K.inverse();
	K = K / K(2,2);

	std::cout << "R:\n" << R << std::endl;
	std::cout << "K:\n" << K << std::endl;
}

void Calibrator::drawOverlay(cv::Mat &frame)
{
	Eigen::MatrixXf x,u,v;
	x = P * X.transpose();

	//std::cout << "image: \n" << x << std::endl;

	u = x.row(0).array() / x.row(2).array();
	v = x.row(1).array() / x.row(2).array();

	//std::cout << "u:\n" << u << std::endl;
	//std::cout << "test: u(2) \n"<< u(2) << std::endl;

	//Mark points
	for(int i = 0; i < X.rows(); i++)
	{
		cv::circle(frame,cv::Point(u(i),v(i)),20,cv::Scalar(255,0,0));
	}

	//Draw lines
	for(int i = 0; i < X.rows() - 1; i++)
	{
		cv::line(frame,cv::Point(u(i),v(i)),cv::Point(u(i+1),v(i+1)),cv::Scalar(0,0,255),5,CV_AA);
	}
}

Eigen::MatrixXf Calibrator::getPMatrix()
{
	return P;
}
