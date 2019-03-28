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
					std::cout << "Initialization complete!\n";
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
	alpha_matte.create(input_img.size(), CV_8UC1);	
	alpha_matte.setTo(BG);
	alpha_matte(roi).setTo(PFG);

	// Initialize fg and bg GMM from initial selection using kmeans
	fg_gmm.models.resize(num_components);
	bg_gmm.models.resize(num_components);

	std::vector<cv::Vec3f> fg_samples, bg_samples;

	for(int x = 0; x < input_img.rows; x++)
		for(int y = 0; y < input_img.cols; y++)
		{
			if(alpha_matte.at<uchar>(x, y) == FG || alpha_matte.at<uchar>(x, y) == PFG)
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
			if(alpha_matte.at<uchar>(x, y) == FG || alpha_matte.at<uchar>(x, y) == PFG)
				pixel_to_component.at<uchar>(x, y) = fg_cluster_indices[i++];
			else
				pixel_to_component.at<uchar>(x, y) = bg_cluster_indices[j++];
		}

	learnGmmParameters();
}

void GrabCut::assignGmmComponents()
{
	cv::Vec3b pixel;
	Eigen::Vector3f eg_pixel;
	Gaussian model;

	double p, p_max, best_component;

	for(int x = 0; x < input_img.rows; x++)
		for(int y = 0; y < input_img.cols; y++)
		{
			pixel = input_img.at<cv::Vec3b>(x, y);
			eg_pixel << pixel[0], pixel[1], pixel[2];

			if(alpha_matte.at<uchar>(x, y) == FG || alpha_matte.at<uchar>(x, y) == PFG)
			{
				p_max = 0;

				for(int i = 0; i < num_components; i++)
				{
					model = fg_gmm.models[i];
					if(model.weight > 0)
					{
						p = 1.0f/sqrt(model.detm_covariance)
							* exp(-0.5f*(eg_pixel - model.mean).transpose() * model.inverse_covariance
									* (eg_pixel - model.mean));
					}

					if(p > p_max)
					{
						p_max = p;
						best_component = i;
					}
				}

				pixel_to_component.at<uchar>(x, y) = best_component;
			}

			else
			{
				p_max = 0;

				for(int i = 0; i < num_components; i++)
				{
					model = bg_gmm.models[i];
					if(model.weight > 0)
					{
						p = 1.0f/sqrt(model.detm_covariance)
							* exp(-0.5f*(eg_pixel - model.mean).transpose() * model.inverse_covariance
									* (eg_pixel - model.mean));
					}

					if(p > p_max)
					{
						p_max = p;
						best_component = i;
					}
				}

				pixel_to_component.at<uchar>(x, y) = best_component;
			}
		}
}

void GrabCut::learnGmmParameters()
{
	// Estimate mean, convariance, component weights
	int component;
	cv::Vec3b pixel;
	Eigen::Vector3f eg_pixel;

	for(int x = 0; x < pixel_to_component.rows; x++)
		for(int y = 0; y < pixel_to_component.cols; y++)
		{
			component = pixel_to_component.at<uchar>(x, y);
			pixel = input_img.at<cv::Vec3b>(x, y);
			eg_pixel << pixel[0], pixel[1], pixel[2];

			if(alpha_matte.at<uchar>(x, y) == FG || alpha_matte.at<uchar>(x, y) == PFG)
			{
				fg_gmm.models[component].mean += eg_pixel;
				fg_gmm.models[component].covariance += eg_pixel * eg_pixel.transpose();
				fg_gmm.models[component].sample_count++;
				fg_gmm.sample_count++;
			}

			else
			{
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
			fg_gmm.models[i].inverse_covariance = fg_gmm.models[i].covariance.inverse();
			fg_gmm.models[i].detm_covariance = fg_gmm.models[i].covariance.determinant();
			fg_gmm.models[i].weight = (double)fg_gmm.models[i].sample_count / fg_gmm.sample_count;
		}

		if(bg_gmm.models[i].sample_count == 0)
			bg_gmm.models[i].weight = 0;

		else
		{
			bg_gmm.models[i].mean /= bg_gmm.models[i].sample_count;
			bg_gmm.models[i].covariance /= bg_gmm.models[i].sample_count;
			bg_gmm.models[i].covariance -= bg_gmm.models[i].mean * bg_gmm.models[i].mean.transpose();
			bg_gmm.models[i].inverse_covariance = bg_gmm.models[i].covariance.inverse();
			bg_gmm.models[i].detm_covariance = bg_gmm.models[i].covariance.determinant();
			bg_gmm.models[i].weight = (double)bg_gmm.models[i].sample_count / bg_gmm.sample_count;
		}
	}
}

double GrabCut::estimateBeta()
{
	// Assume 8 neighbors
	double beta = 0;
	cv::Vec3b pixel, diff;
	for(int x = 0; x < input_img.rows; x++)
		for(int y = 0; y < input_img.cols; y++)
		{
			pixel = input_img.at<cv::Vec3b>(x, y);

			if(x > 0)
			{
				diff = pixel - input_img.at<cv::Vec3b>(x - 1, y);
				beta += diff.dot(diff);
			}

			if(y > 0)
			{
				diff = pixel - input_img.at<cv::Vec3b>(x, y - 1);
				beta += diff.dot(diff);
			}

			if(x>0 && y>0)
			{
				diff = pixel - input_img.at<cv::Vec3b>(x - 1, y - 1);
				beta += diff.dot(diff);
			}

			if(x>0 && y<input_img.cols-1)
			{
				diff = pixel - input_img.at<cv::Vec3b>(x + 1, y - 1);
				beta += diff.dot(diff);
			}
		}

	beta = (2.0f*beta) / (4*input_img.cols*input_img.rows - 3*input_img.cols - 3*input_img.rows +2);
	beta = 1.0f/beta;

	return beta;
}


void GrabCut::estimateNeighborWeights(const double &beta, const double &gamma, cv::Mat &left_weights,
		cv::Mat &up_weights, cv::Mat &up_left_weights, cv::Mat &up_right_weights)
{
	cv::Vec3b pixel, diff;

	for(int x = 0; x < input_img.rows; x++)
		for(int y = 0; y < input_img.cols; y++)
		{
			pixel = input_img.at<cv::Vec3b>(x, y);
			if(x > 0)
			{
				diff = pixel - input_img.at<cv::Vec3b>(x - 1, y);
				left_weights.at<double>(x, y) = gamma * exp(-beta*diff.dot(diff));
			}

			if(y > 0)
			{
				diff = pixel - input_img.at<cv::Vec3b>(x, y - 1);
				up_weights.at<double>(x, y) = gamma * exp(-beta*diff.dot(diff));
			}

			if(x>0 && y>0)
			{
				diff = pixel - input_img.at<cv::Vec3b>(x - 1, y - 1);
				up_left_weights.at<double>(x, y) = (gamma/sqrt(2)) * exp(-beta*diff.dot(diff));
			}

			if(x>0 && y<input_img.cols-1)
			{
				diff = pixel - input_img.at<cv::Vec3b>(x + 1, y - 1);
				up_right_weights.at<double>(x, y) = (gamma/sqrt(2)) * exp(-beta*diff.dot(diff));
			}
		}
}


void GrabCut::constructGraph(const double &lambda, const cv::Mat &left_weights,
		const cv::Mat &up_weights, const cv::Mat &up_left_weights,
		const cv::Mat &up_right_weights, GraphType *graph)
{
	cv::Vec3b pixel;
	Eigen::Vector3f eg_pixel;
	int node_id;
	Gaussian model;
	double source_weight, sink_weight, n_weight;

	for(int x = 0; x < input_img.rows; x++)
		for(int y = 0; y < input_img.cols; y++)
		{
			pixel = input_img.at<cv::Vec3b>(x,y);
			eg_pixel << pixel[0], pixel[1], pixel[2];
			node_id = graph->add_node();
			source_weight = sink_weight = n_weight = 0;

			if(alpha_matte.at<uchar>(x, y) == FG)
			{
				source_weight = lambda;
				sink_weight = 0;
			}

			else if(alpha_matte.at<uchar>(x, y) == BG)
			{
				source_weight = 0;
				sink_weight = lambda;
			}

			else if(alpha_matte.at<uchar>(x, y) == PFG || alpha_matte.at<uchar>(x,y) == PBG)
			{
				for(int i = 0; i < num_components; i++)
				{
					model = fg_gmm.models[i];
					if(model.weight > 0)
					{
						//double temp_product = (eg_pixel - model.mean).transpose() * model.inverse_covariance * (eg_pixel - model.mean);
						//std::cout << temp_product << std::endl;
						//double sqrt_term = 1.0f/sqrt(model.detm_covariance);
						//std::cout << "sqrt_term: " << sqrt_term << std::endl;
						//double exp_term = exp(0.5f*temp_product);
						//std::cout << "exp: " << exp_term << std::endl;

						//sink_weight = sink_weight + model.weight * sqrt_term * exp_term;
						
						sink_weight += model.weight * 1.0f/sqrt(model.detm_covariance)
							* exp(-0.5f*(eg_pixel - model.mean).transpose() * model.inverse_covariance * (eg_pixel - model.mean));
					}
					
					model = bg_gmm.models[i];
					if(model.weight > 0)
					{
						source_weight += model.weight * 1.0f/sqrt(model.detm_covariance)
							* exp(-0.5f*(eg_pixel - model.mean).transpose() * model.inverse_covariance * (eg_pixel - model.mean));
					}
				}

				//std::cout << source_weight << " " << sink_weight << std::endl;
				source_weight = -log(source_weight);
				sink_weight = -log(sink_weight);
			}

			graph->add_tweights(node_id, source_weight, sink_weight);

			if(x > 0)
			{
				n_weight = left_weights.at<double>(x, y);
				graph->add_edge(node_id, node_id - input_img.cols, n_weight, n_weight);
			}

			if(y > 0)
			{
				n_weight = up_weights.at<double>(x, y);
				graph->add_edge(node_id, node_id - 1, n_weight, n_weight);
			}

			if(x>0 && y>0)
			{
				n_weight = up_left_weights.at<double>(x, y);
				graph->add_edge(node_id, node_id - input_img.cols - 1, n_weight, n_weight);
			}

			if(x>0 && y<input_img.cols-1)
			{
				n_weight = up_right_weights.at<double>(x, y);
				graph->add_edge(node_id, node_id -input_img.cols + 1, n_weight, n_weight);
			}
		}
}

void GrabCut::estimateSegmentation(GraphType *graph)
{
	int flow = graph->maxflow();
	std::cout << flow << std::endl;

	for(int x = 0; x < alpha_matte.rows; x++)
		for(int y = 0; y < alpha_matte.cols; y++)
		{
			if(alpha_matte.at<uchar>(x, y) == PBG || alpha_matte.at<uchar>(x, y) == PFG)
			{
				if(graph->what_segment(x*alpha_matte.cols + y) == GraphType::SOURCE)
					alpha_matte.at<uchar>(x, y) = PFG;

				else
					alpha_matte.at<uchar>(x, y) = PBG;
			}
		}
}

void GrabCut::displayResult()
{
	cv::Mat result_image = cv::Mat::zeros(input_img.size(), CV_8UC3);

	for(int i = 0; i < input_img.rows; i++)
		for(int j = 0; j < input_img.cols; j++)
			if(alpha_matte.at<uchar>(i, j) == PFG || alpha_matte.at<uchar>(i, j) == FG)
				result_image.at<cv::Vec3b>(i, j) = input_img.at<cv::Vec3b>(i, j);

	cv::namedWindow("Result image", CV_WINDOW_NORMAL);
	cv::imshow("Result image", result_image);
	cv::waitKey(0);
}

void GrabCut::runGrabCut()
{
	int num_nodes = input_img.cols * input_img.rows;
	int num_edges = 4*input_img.cols*input_img.rows - 3*input_img.cols -3*input_img.rows + 2;

	double beta = estimateBeta();
	double gamma = 10;
	double lambda = 100;

	cv::Mat left_weights = cv::Mat::zeros(input_img.size(), CV_64FC1);
	cv::Mat up_weights = cv::Mat::zeros(input_img.size(), CV_64FC1);
	cv::Mat up_left_weights = cv::Mat::zeros(input_img.size(), CV_64FC1);
	cv::Mat up_right_weights = cv::Mat::zeros(input_img.size(), CV_64FC1);

	estimateNeighborWeights(beta, gamma, left_weights, up_weights, up_left_weights, up_right_weights);

	//std::cout << left_weights << std::endl;

	GraphType *graph = new GraphType(num_nodes, num_edges);
	//assignGmmComponents();
	//learnGmmParameters();
	constructGraph(lambda, left_weights, up_weights, up_left_weights, up_right_weights, graph);
	estimateSegmentation(graph);

	for(int i = 1; i < 10; i++)
	{
		assignGmmComponents();
		learnGmmParameters();
		GraphType *graph = new GraphType(num_nodes, num_edges);
		constructGraph(lambda, left_weights, up_weights, up_left_weights, up_right_weights, graph);
		estimateSegmentation(graph);
		displayResult();
	}

}
