#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Core>

struct Gaussian
{
	Eigen::Vector3f mean;
	Eigen::Matrix3f covariance;
	float weight = 0;
	int sample_count = 0;
};

struct GMM
{
	std::vector<Gaussian> models;
	int sample_count = 0;
};
