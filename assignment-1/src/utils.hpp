#pragma once

#include <iostream>
#include <string>
#include <Eigen/Core>
#include <fstream>
#include <random>
#include <algorithm>

namespace utils
{
	int getNumberofLines(const std::string &file_name);

	void loadMatrix(const std::string &file_name, const int &nrows, const int &ncols, Eigen::MatrixXf &mat);

	std::vector<int> generateRandomVector(const int &lrange, const int &urange, const int &size);
}

