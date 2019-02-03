#include "calibrator.hpp"
#include <stdlib.h>

Calibrator::Calibrator(const Eigen::MatrixXf &points_3d, const Eigen::MatrixXf &points_2d)
{
	x = points_2d;
	x.conservativeResize(x.rows(),x.cols()+1);
	x.col(x.cols()-1).setOnes();

	X = points_3d;
	X = X * 0.01; // assumes points are in cm, converting to m
	X.conservativeResize(X.rows(),X.cols()+1);
	X.col(X.cols()-1).setOnes();
}

void Calibrator::calibrateByDlt(const std::vector<int> &sample_indices)
{
	if(sample_indices.size() < 6)
	{
		std::cout << "Cannot run DLT! Require at least 6 correspondences.\n";
		return;
	}

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
			x_samples.row(j++) = x.row(i);
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

	// Estimate p
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(M,Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXf V;
	V = svd.matrixV();
	p = V.col(V.cols()-1);

	// Build P
 	P.resize(3,4);
	P.row(0) = p.segment(0,4);
	P.row(1) = p.segment(4,4);
	P.row(2) = p.segment(8,4);
	std::cout << "P:\n" << P << std::endl;
	std::cout << "Average reprojection error:\n" << calcAvgReprojectionError() << std::endl;
}

void Calibrator::calibrateByDltRansac(const float &dist_threshold, const float &inlier_ratio, std::vector<int> &inlier_indices)
{	
	std::cout << "Running RANSAC...." << std::endl;
	std::vector<int> largest_support;

	for(int n = 0; n < 500; n++)
	{
		std::cout << "\n\nIteration #" << n+1 << std::endl;
		std::vector<int> sample_indices = utils::generateRandomVector(0,X.rows()-1,6);
		calibrateByDlt(sample_indices);
	
		for(int i = 0; i < X.rows(); i++)
		{
			float dist = calcReprojectionError(X.row(i),x.row(i));
			if(dist < dist_threshold)
				inlier_indices.push_back(i); 
		}

		if(inlier_indices.size() >= inlier_ratio * X.rows())
		{
			std::cout << "Found a model!\n" << "Number of inliers: " << inlier_indices.size() << std::endl;
			std::cout << "Inliers: ";
			for(int i : inlier_indices)
				std::cout << i << " ";

			std::cout << "\n";
			calibrateByDlt(inlier_indices);
			return;
		}
	
		else
		{
			if(largest_support.size() <= inlier_indices.size())
				largest_support = inlier_indices;

			inlier_indices.clear();
		}
	}

	if(largest_support.size() >= 6)
	{
		std::cout << "Could not find a model according to threshold!\nSo using largest inlier set instead." << std::endl;
		calibrateByDlt(largest_support);
		inlier_indices = largest_support;
		std::cout << "Number of inliers: " << inlier_indices.size() << std::endl;
		for(int i : inlier_indices)
			std::cout << i << " ";

		std::cout << "\n";
	}

	else
		std::cout << "Could not find a model!" << std::endl;
}

float Calibrator::calcReprojectionError(const Eigen::Vector4f &pt_3d, const Eigen::Vector3f &pt_img)
{
	Eigen::Vector3f est_pt_img = P * pt_3d;
	est_pt_img = est_pt_img / est_pt_img(2);
	float error = (est_pt_img - pt_img).squaredNorm();

	return error;
}


float Calibrator::calcSetReprojectionError(const Eigen::MatrixXf &pts_3d, const Eigen::MatrixXf &pts_img)
{
	Eigen::MatrixXf est_pts_img = P * pts_3d.transpose();
	Eigen::MatrixXf scale = est_pts_img.row(2).replicate(3,1);
	est_pts_img = est_pts_img.array() / scale.array();

	float total_error = 0;

	for(int i = 0; i < pts_3d.rows(); i++)
	{
		total_error += (est_pts_img.col(i) - pts_img.row(i).transpose()).squaredNorm();
	}

	float avg_error = total_error/pts_3d.rows();
	return avg_error;
}

float Calibrator::calcAvgReprojectionError()
{
	Eigen::MatrixXf est_pts_img = P * X.transpose();
	Eigen::MatrixXf scale = est_pts_img.row(2).replicate(3,1);
	est_pts_img = est_pts_img.array() / scale.array();

	float total_error = 0;

	for(int i = 0; i < X.rows(); i++)
	{
		total_error += (est_pts_img.col(i) - x.row(i).transpose()).squaredNorm();
	}

	float avg_error = total_error/X.rows();
	return avg_error;
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

void Calibrator::drawOverlay(const std::vector<int> &sample_indices, cv::Mat &frame)
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
			x_samples.row(j++) = x.row(i);
		}
	}

	Eigen::MatrixXf est_x,est_u,est_v;
	est_x = P * X_samples.transpose();

	//std::cout << "image: \n" << x << std::endl;

	est_u = est_x.row(0).array() / est_x.row(2).array();
	est_v = est_x.row(1).array() / est_x.row(2).array();

	//std::cout << "u:\n" << u << std::endl;
	//std::cout << "test: u(2) \n"<< u(2) << std::endl;

	//Mark points
	for(int i = 0; i < x_samples.rows(); i++)
	{
		cv::circle(frame,cv::Point(x_samples(i,0),x_samples(i,1)),20,cv::Scalar(0,0,255),-1,CV_AA);
		cv::circle(frame,cv::Point(est_u(i),est_v(i)),20,cv::Scalar(255,0,0),-1,CV_AA);
	}

	//Draw lines
	for(int i = 0; i < x_samples.rows() - 1; i++)
	{
		cv::line(frame,cv::Point(x_samples(i,0),x_samples(i,1)),cv::Point(x_samples(i+1,0),x_samples(i+1,1)),cv::Scalar(0,0,255),5,CV_AA);
		cv::line(frame,cv::Point(est_u(i),est_v(i)),cv::Point(est_u(i+1),est_v(i+1)),cv::Scalar(255,0,0),5,CV_AA);
	}
}

Eigen::MatrixXf Calibrator::getPMatrix()
{
	return P;
}
