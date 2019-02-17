#include "panaroma.h"

/** Starts the panaroma program
 * Takes as input the image filenames to load, and desired output filename for mosaic
 * Constructs a Panaroma object and calls its run method
*/

int main(int argc, char *argv[])
{
    if(argc < 4)
    {
        std::cout << "Incorrect input format!\n Correct usage: ./run <image1-path> <image2-path> [<image3-path> ...]  <ouput-mosaic-path> \n";
        return -1;
    }

    std::vector<cv::Mat> images;

    for(int i = 1; i < argc - 1; i++)
    {
        cv::Mat img = cv::imread(argv[i],1);
        images.push_back(img);
    }

    std::cout << images.size() << " images loaded!\n";

    Panaroma panaroma(images,argv[argc-1]);
    panaroma.run(1, 0.8);

    return 0;
}
