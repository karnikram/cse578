#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class GrabCut
{
	public:
		GrabCut();
		void setInputImage(const cv::Mat &input_img);
		void showImage();
		void captureMouseClick(int event, int x, int y, int flags, void *userdata);
		void initializeGMM();
		void learnGMM();
		void runGraphCut();
		enum{NOT_SET = 0, IN_PROCESS = 1, SET = 2};

	private:
		cv::Mat input_img, roi_image;
		cv::Rect roi;
		int inter_count;
		uchar roi_state;
};
