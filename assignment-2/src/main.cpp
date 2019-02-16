// Accept image paths via cli
// Load images
// Perform feature matching - in which order?
// homography estimation

#include "panaroma.h"

int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        std::cout << "Incorrect input format!\n Correct usage: ./main <image1-path> <image2-path> [image3-path] ...\n";
        return -1;
    }

    std::vector<cv::Mat> images;

    for(int i = 1; i < argc; i++)
    {
        cv::Mat img = cv::imread(argv[i],1);
        images.push_back(img);
    }

    std::cout << images.size() << " images loaded!\n";

    Panaroma panaroma(images);
    panaroma.run(3, 0.995);

    return 0;
}
