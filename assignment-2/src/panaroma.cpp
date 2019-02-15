#include "panaroma.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/LU>

#include <algorithm>

Panaroma::Panaroma(std::vector<cv::Mat> images)
{
	this->images = images;
}

void Panaroma::generateMatches(const cv::Mat &img1, const cv::Mat &img2,
    	Eigen::MatrixXf &X1, Eigen::MatrixXf &X2)
{
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    detector->detectAndCompute(images[0], cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(images[1], cv::noArray(), keypoints2, descriptors2);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good_matches;
    for(size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    cv::Mat img_matches;
    cv::drawMatches(images[0], keypoints1, images[1], keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Good Matches", img_matches);

	X1.resize(good_matches.size(),2);
	X2.resize(good_matches.size(),2);

	size_t j = 0;
	for(cv::DMatch match : good_matches)
	{
		cv::Point2f pt = keypoints1[match.queryIdx].pt;
		X1(j,0) = pt.x;
		X1(j,1) = pt.y;

		pt = keypoints2[match.trainIdx].pt;
		X2(j,0) = pt.x;
		X2(j++,1) = pt.y;
	}

	X1.conservativeResize(X1.rows(),X1.cols()+1);
	X2.conservativeResize(X1.rows(),X2.cols()+1);

	X1.col(X1.cols()-1).setOnes();
	X2.col(X2.cols()-1).setOnes();
}

void Panaroma::sampleFromX1X2(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2,
		const std::vector<int> &sample_indices, Eigen::MatrixXf &sample_X1,
		Eigen::MatrixXf &sample_X2)
{
	sample_X1.resize(sample_indices.size(),3);
	sample_X2.resize(sample_indices.size(),3);

	int j = 0;
	for(int i : sample_indices)
	{
		sample_X1.row(j) = X1.row(i);
		sample_X2.row(j++) = X2.row(i);
	}
}

void Panaroma::estimateHomography(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2,
    	Eigen::MatrixXf &H)
{
	Eigen::MatrixXf A(2*X1.rows(), 9);
	Eigen::ArrayXf h(9);

	for(int i = 0, j = 0; i < A.rows(); i+=2, j++)
	{
		A.row(i) << 0, 0, 0, -1 * X2(j,2) * X1.row(j), X2(j,1) * X1.row(j);
		A.row(i+1) << X2(j,2) * X1.row(j), 0, 0, 0, -1 * X2(j,0) * X1.row(j);
	}

	//std::cout << "A:\n" << A << std::endl;

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A,Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXf V;
	V = svd.matrixV();
	h = V.col(V.cols()-1);

	H.resize(3,3);
	H.row(0) = h.segment(0,3);
	H.row(1) = h.segment(3,3);
	H.row(2) = h.segment(6,3);

	//std::cout << "h:\n" << h << std::endl;
	std::cout << "H:\n" << H << std::endl;

	//H = H / H(2,2);
	//std::cout << "Normalized H:\n" << H << std::endl;
}

void Panaroma::estimateRansacHomography(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2,
		const float &dist_threshold, const float &ratio_threshold, Eigen::MatrixXf &H,
		std::vector<int> &inlier_indices)
{
	std::cout << "Estimating Homography within RANSAC..." << std::endl;
	Eigen::MatrixXf sample_X1, sample_X2;
	std::vector<int> largest_support;

	for(size_t i = 0; i < 500; i++)
	{
		std::cout << "\n\nIteration#" << i+1 << std::endl;
		std::vector<int> sample_indices = utils::generateRandomVector(0,X1.rows()-1,4);
		sampleFromX1X2(X1,X2,sample_indices,sample_X1,sample_X2);
		
		estimateHomography(sample_X1,sample_X2,H);

		for(size_t j = 0; j < X1.rows(); j++)
		{
			if(std::find(sample_indices.begin(),sample_indices.end(),j) != sample_indices.end())
				continue;

			Eigen::Vector3f px2	= H * (X1.row(j).transpose());
			px2 = px2 / px2(2);
			Eigen::Vector3f x2 = X2.row(j);

			if( ((px2 - x2).squaredNorm()) < dist_threshold )
				inlier_indices.push_back(j);
		}

		if(inlier_indices.size() >= X1.rows() * ratio_threshold)
		{
			std::cout << "\nFound a model!\nNumber of inliers: " << inlier_indices.size() << std::endl;
			inlier_indices.insert(inlier_indices.end(),sample_indices.begin(),sample_indices.end());
			sampleFromX1X2(X1,X2,inlier_indices,sample_X1,sample_X2);
			estimateHomography(sample_X1,sample_X2,H);

			std::cout << "Original number of SIFT matches: " << X1.rows() << std::endl;
			return;
		}

		else
		{
			if(largest_support.size() < inlier_indices.size())
			{
				largest_support = inlier_indices;
				largest_support.insert(largest_support.end(),sample_indices.begin(),sample_indices.end());
			}

			inlier_indices.clear();
		}
	}

	if(largest_support.size() >= 4)
	{
		std::cout << "\nCould not find a model according to threshold!\nSo using largest inlier set instead." << std::endl;
		std::cout << "Number of inliers: " << largest_support.size() << std::endl;
		sampleFromX1X2(X1,X2,largest_support,sample_X1,sample_X2);
		estimateHomography(sample_X1,sample_X2,H);
		inlier_indices = largest_support;
		std::cout << "Original number of SIFT matches: " << X1.rows() << std::endl;
	}

	else
		std::cout << "Could not find a model!" << std::endl;
}

void Panaroma::warpImage(const cv::Mat &src_img, const Eigen::Matrix3f &H, cv::Mat &dst_img)
{
	cv::Mat map_x, map_y;
	dst_img = cv::Mat::zeros(src_img.size(),src_img.type());

	map_x.create(src_img.size(),CV_32FC1);
	map_y.create(src_img.size(),CV_32FC1);
	
	for(int i = 0; i < dst_img.rows; i++)
		for(int j = 0; j < dst_img.cols; j++)
		{
			Eigen::Vector3f x(j,i,1);
			Eigen::Vector3f px = H.inverse() * x;
			px = px/px(2);

			map_x.at<float>(i,j) = px(0);
			map_y.at<float>(i,j) = px(1);
		}

	cv::remap(src_img,dst_img,map_x,map_y,CV_INTER_LINEAR);
}

void Panaroma::run(const float &dist_threshold, const float &ratio_threshold)
{
	Eigen::MatrixXf X1,X2;
    generateMatches(images[0],images[1],X1,X2);

    Eigen::MatrixXf H;
    std::vector<int> inlier_indices;
    estimateRansacHomography(X1,X2,dist_threshold,ratio_threshold,H,inlier_indices);

    cv::Mat cv_H;
    eigen2cv(H, cv_H);

    cv::Mat testImage;
    //cv::warpPerspective(images[1],testImage,cv_H,images[0].size());

    warpImage(images[1],H,testImage);
    cv::imshow("warped",testImage);
    cv::waitKey(0);
}
