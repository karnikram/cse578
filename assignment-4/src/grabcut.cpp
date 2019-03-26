#include "grabcut.h"
#include <opencv2/highgui.hpp>

GrabCut::GrabCut()
{
}

void GrabCut::setInputImage(const cv::Mat &input_img)
{
	this->input_img = input_img;
}

uchar GrabCut::getRoiState()
{
	return this->roi_state;
}

void GrabCut::showInputImage()
{
	cv::Mat temp_img;
	input_img.copyTo(temp_img);

	if(roi_state == IN_PROCESS || roi_state == SET)
		cv::rectangle(temp_img,cv::Point(roi.x,roi.y),cv::Point(roi.x + roi.width,
					roi.y + roi.height),cv::Scalar(0,0,255),2);

	cv::imshow("Input image",temp_img);
}

void GrabCut::captureMouseClick(int event, int x, int y, int flags, void *userdata)
{
	switch(event)
	{
		case CV_EVENT_LBUTTONDOWN:
			{
				if(roi_state == NOT_SET)
				{
					roi_state = IN_PROCESS;
					roi = cv::Rect(x,y,1,1);
				}
			}
			break;

		case CV_EVENT_LBUTTONUP:
			{
				if(roi_state == IN_PROCESS)
				{
					roi = cv::Rect(cv::Point(roi.x,roi.y),cv::Point(x,y));
					roi_state = SET;
					showInputImage();
					initializeModel();
				}
			}
			break;

		case CV_EVENT_MOUSEMOVE:
			{
				if(roi_state == IN_PROCESS)
				{
					roi = cv::Rect(cv::Point(roi.x,roi.y),cv::Point(x,y));
					showInputImage();
				}
			}
			break;
	}
}

void GrabCut::initializeModel()
{
	// Assign initial opacity values based on selected background region
	alpha_matte = cv::Mat::zeros(input_img.size(), CV_8UC1);	
	alpha_matte(roi).setTo(FG);

	// Initialize fg and bg GMM from initial selection using kmeans
	fg_gmm.models.resize(num_components);
	bg_gmm.models.resize(num_components);

	std::vector<cv::Vec3f> fg_samples, bg_samples;

	for(int x = 0; x < input_img.rows; x++)
		for(int y = 0; y < input_img.cols; y++)
		{
			if(roi.contains(cv::Point2f(x,y)))
				fg_samples.push_back((cv::Vec3f)input_img.at<cv::Vec3b>(x, y));
			else
				bg_samples.push_back((cv::Vec3f)input_img.at<cv::Vec3b>(x, y));
		}

	std::vector<int> fg_cluster_indices, bg_cluster_indices;
	cv::kmeans(fg_samples, num_components, fg_cluster_indices, cv::TermCriteria(CV_TERMCRIT_ITER, 10, 0.0), 0, cv::KMEANS_PP_CENTERS);
	cv::kmeans(bg_samples, num_components, bg_cluster_indices, cv::TermCriteria(CV_TERMCRIT_ITER, 10, 0.0), 0, cv::KMEANS_PP_CENTERS);

	int i = 0, j = 0;
	pixel_to_component.create(input_img.size(), CV_8UC1);
	for(int x = 0; x < pixel_to_component.rows; x++)
		for(int y = 0; y < pixel_to_component.cols; y++)
		{
			if(roi.contains(cv::Point2f(x, y)))
				pixel_to_component.at<uchar>(x, y) = fg_cluster_indices[i++];
			else
				pixel_to_component.at<uchar>(x, y) = bg_cluster_indices[j++];
		}

	learnGmmParameters();
}

void GrabCut::assignGmmComponents()
{
}

void GrabCut::learnGmmParameters()
{
	// Estimate mean, convariance, component weights
	int component;
	cv::Vec3b pixel;
	Eigen::Vector3f eg_pixel;

	for(int i = 0; i < pixel_to_component.rows; i++)
		for(int j = 0; j < pixel_to_component.cols; j++)
		{
			if(alpha_matte.at<uchar>(i, j) == FG)
			{
				component = pixel_to_component.at<uchar>(i, j);
				pixel = input_img.at<cv::Vec3b>(i, j);
				eg_pixel << pixel[0], pixel[1], pixel[2];
				fg_gmm.models[component].mean += eg_pixel;
				fg_gmm.models[component].covariance += eg_pixel * eg_pixel.transpose();
				fg_gmm.models[component].sample_count++;
				fg_gmm.sample_count++;
			}

			else
			{
				component = pixel_to_component.at<uchar>(i, j);
				pixel = input_img.at<cv::Vec3b>(i, j);
				eg_pixel << pixel[0], pixel[1], pixel[2];
				bg_gmm.models[component].mean += eg_pixel;
				bg_gmm.models[component].covariance += eg_pixel * eg_pixel.transpose();
				bg_gmm.models[component].sample_count++;
				bg_gmm.sample_count++;
			}
		}
	
	for(int i = 0; i < num_components; i++)
	{
		if(fg_gmm.models[i].sample_count == 0)
			fg_gmm.models[i].weight = 0;

		else
		{
			fg_gmm.models[i].mean /= fg_gmm.models[i].sample_count;
			fg_gmm.models[i].covariance /= fg_gmm.models[i].sample_count;
			fg_gmm.models[i].covariance -= fg_gmm.models[i].mean * fg_gmm.models[i].mean.transpose();
			fg_gmm.models[i].weight = fg_gmm.models[i].sample_count / fg_gmm.sample_count;
		}

		if(bg_gmm.models[i].sample_count == 0)
			bg_gmm.models[i].weight = 0;

		else
		{
			bg_gmm.models[i].mean /= bg_gmm.models[i].sample_count;
			bg_gmm.models[i].covariance /= bg_gmm.models[i].sample_count;
			bg_gmm.models[i].covariance -= bg_gmm.models[i].mean * bg_gmm.models[i].mean.transpose();
			bg_gmm.models[i].weight = bg_gmm.models[i].sample_count / bg_gmm.sample_count;
		}
	}

	std::cout << "Done!\n";
}

void GrabCut::constructGraph()
{
	for(int x = 0; x < input_img.rows; x++)
		for(int y = 0; y < input_img.cols; y++)
		{
			cv::Vec3b px_color = input_img.at<cv::Vec3b>(x,y);
		}
}

void GrabCut::estimateSegmentation()
{
}
