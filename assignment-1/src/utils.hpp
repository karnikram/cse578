#pragma once

#include <iostream>
#include <string>
#include <Eigen/Core>
#include <fstream>

namespace utils
{
	int getNumberofLines(const std::string &file_name);

	void loadMatrix(const std::string &file_name, const int &nrows, const int &ncols, Eigen::MatrixXf &mat);
}

