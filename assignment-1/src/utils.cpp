#include "utils.hpp"

int utils::getNumberofLines(const std::string &file_name)
{
	std::ifstream file(file_name);
	std::string line;
	int num_lines = 0;

	while(std::getline(file,line))
		num_lines++;

	file.close();

	return num_lines;
}

void utils::loadMatrix(const std::string &file_name, const int &nrows, const int &ncols, Eigen::MatrixXf &mat)
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

std::vector<int> utils::generateRandomVector(const int &lrange, const int &urange, const int &size)
{
    static std::uniform_int_distribution<int> distribution(lrange,urange);
    static std::default_random_engine generator;

    std::vector<int> data(size);
    std::generate(data.begin(), data.end(), []() { return distribution(generator); });
    return data;
}
