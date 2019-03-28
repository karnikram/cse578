#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

struct Gaussian
{
	Eigen::Vector3f mean;
	Eigen::Matrix3f covariance;
	Eigen::Matrix3f inverse_covariance;
	double detm_covariance;
	double weight = 0;
	int sample_count = 0;
};

struct GMM
{
	std::vector<Gaussian> models;
	int sample_count = 0;
};
