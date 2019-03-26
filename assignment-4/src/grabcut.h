#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "gmm.h"

class GrabCut
{
	public:
		GrabCut();
		void setInputImage(const cv::Mat &input_img);
		void showInputImage();
		void captureMouseClick(int event, int x, int y, int flags, void *userdata);
		uchar getRoiState();
		void initializeModel();
		void assignGmmComponents();
		void learnGmmParameters();
		void constructGraph();
		void estimateSegmentation();
		enum{NOT_SET = 0, IN_PROCESS = 1, SET = 2};
		enum{FG = 1, BG = 0};

	private:
		cv::Mat input_img, roi_image;
		cv::Rect roi;
		uchar roi_state;
		cv::Mat alpha_matte;
		int num_components = 5;
		GMM bg_gmm, fg_gmm;
		cv::Mat pixel_to_component;
};
