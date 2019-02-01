#pragma once

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/LU>

#include "utils.hpp"

class Calibrator
{
	public: 
		Calibrator(const Eigen::MatrixXf &points_3d, const Eigen::MatrixXf &points_2d);
		void calibrateByDlt(const std::vector<int> &sample_indices);
		void calibrateByDltRansac(const float &dist_threshold);
		float calcReprojectionError(const Eigen::Vector4f &X, const Eigen::Vector3f &x);
		void decomposePMatrix(Eigen::MatrixXf &K, Eigen::MatrixXf &R, Eigen::MatrixXf &c);
		void drawOverlay(cv::Mat &frame);
		Eigen::MatrixXf getPMatrix();

	private:
		Eigen::MatrixXf X;
		Eigen::MatrixXf x;
		Eigen::MatrixXf P;
};
