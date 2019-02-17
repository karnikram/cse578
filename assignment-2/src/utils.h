#pragma once

#include <iostream>
#include <string>
#include <Eigen/Core>
#include <fstream>
#include <random>
#include <algorithm>

namespace utils
{
	/** Generates a vector containing uniformally distributed random values
	 * \param lrange The lower limit
	 * \param uprange The upper limit 
	 * \param size The size of the vector
	*/
	std::vector<int> generateRandomVector(const int &lrange, const int &urange, const int &size);
}

