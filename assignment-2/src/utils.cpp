#include "utils.hpp"

std::vector<int> utils::generateRandomVector(const int &lrange, const int &urange, const int &size)
{
    static std::uniform_int_distribution<int> distribution(lrange,urange);
    static std::default_random_engine generator;

    std::vector<int> data(size);
    std::generate(data.begin(), data.end(), []() { return distribution(generator); });
    return data;
}
