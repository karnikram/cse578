#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "gmm.h"
#include "graph.h"

typedef Graph<int, int, int> GraphType;

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

		double estimateBeta();

		void estimateNeighborWeights(const double &beta, const double &gamma,
				cv::Mat &left_weights, cv::Mat &up_weights, cv::Mat &up_left_weights,
				cv::Mat &up_right_weights);

		void constructGraph(const double &lambda, const cv::Mat &left_weights,
		        const cv::Mat &up_weights, const cv::Mat &up_left_weights,
                const cv::Mat &up_right_weights, GraphType *graph);

		void estimateSegmentation(GraphType *graph);

		void displayResult();

		void runGrabCut();

		enum{NOT_SET = 0, IN_PROCESS = 1, SET = 2};

		enum{BG = 0, FG = 1, PBG = 2, PFG = 3};

	private:
		cv::Mat input_img, roi_image;
		cv::Rect roi;
		uchar roi_state;
		cv::Mat alpha_matte;
		int num_components = 5;
		GMM bg_gmm, fg_gmm;
		cv::Mat pixel_to_component;
};
