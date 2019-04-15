#include "utils.h"
#include <chrono>

std::vector<int> utils::generateRandomVector(const int &lrange, const int &urange, const int &size)
{
    
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> distribution(lrange,urange);

    std::vector<int> data(size);

    for(int i = 0; i < size; i++)
    {
       data[i] = distribution(generator);
    }

    return data;
}
