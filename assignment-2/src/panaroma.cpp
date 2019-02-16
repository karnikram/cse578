#include "panaroma.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <fstream>

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
	sample_X1.resize(0,0);
	sample_X2.resize(0,0);

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
	
	cv::Mat cv_A,U,S,Vt;
	eigen2cv(A,cv_A);
	cv::SVD::compute(cv_A,U,S,Vt);

	//Eigen::JacobiSVD<Eigen::MatrixXf> svd(A,Eigen::ComputeThinU | Eigen::ComputeThinV);
	//Eigen::JacobiSVD<Eigen::MatrixXf> svd(A,Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXf V,tempV;
	cv2eigen(Vt,tempV);
	V = tempV.transpose();

	//std::cout << "V:\n" << V << std::endl;
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

float Panaroma::calcReprojectionError(const Eigen::MatrixXf &x1, const Eigen::MatrixXf &x2, const Eigen::MatrixXf &H)
{
	Eigen::Vector3f px2;
	px2	= H * x1;
	px2 = px2 / px2(2);

	//std::cout << x1.transpose() << "   " << x2.transpose() << "   " <<  px2.transpose() << std::endl;

	return (px2 - x2).squaredNorm();
}

float Panaroma::calcAvgReprojectionError(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2, const Eigen::MatrixXf &H)
{
	float avg = 0, error;
	Eigen::Vector3f x1, x2, px2;

	for(size_t j = 0; j < X1.rows(); j++)
	{
		x1 = X1.row(j);
		x2 = X2.row(j);
		error = calcReprojectionError(x1,x2,H);
		//std::cout << error << std::endl;
		avg += error;
	}

	return avg/X1.rows();
}

void Panaroma::estimateRansacHomography(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2,
		const float &dist_threshold, const float &ratio_threshold, Eigen::MatrixXf &H,
		std::vector<int> &inlier_indices)
{
	std::cout << "Estimating Homography within RANSAC..." << std::endl;
	Eigen::MatrixXf sample_X1, sample_X2;
	Eigen::Vector3f x1,x2;
	std::vector<int> largest_support;
	float inlier_avg = 0;

	for(size_t i = 0; i < 2000; i++)
	{
		std::cout << "\n\nIteration#" << i+1 << std::endl;
		std::vector<int> sample_indices = utils::generateRandomVector(0,X1.rows()-1,4);

		sampleFromX1X2(X1,X2,sample_indices,sample_X1,sample_X2);
		
		estimateHomography(sample_X1,sample_X2,H);

		//std::ofstream file1("file1.txt",std::ofstream::trunc);

		for(size_t j = 0; j < X1.rows(); j++)
		{
			if(std::find(sample_indices.begin(),sample_indices.end(),j) != sample_indices.end())
				continue;

			x1 = X1.row(j);
			x2 = X2.row(j);

			if( calcReprojectionError(x1,x2,H) <= dist_threshold )
			{
				inlier_indices.push_back(j);
				inlier_avg += calcReprojectionError(x1,x2,H);
				//file1 << x1.transpose() << "   " << x2.transpose() << std::endl;
			}
		}

		for(size_t k = 0; k < sample_X1.rows(); k++)
		{
			//file1 << sample_X1.row(k) << "   " << sample_X2.row(k) << std::endl;
		}

		//file1.close();
		std::cout << "\nInlier avg: " << inlier_avg / inlier_indices.size() << std::endl;

		//if((inlier_avg/inlier_indices.size()) < 1 && inlier_indices.size() >= 1)
		//{
		//	std::cout << "Inlier avg < 1\n";
		//	return;
		//}

		inlier_avg = 0;

		if(inlier_indices.size() >= X1.rows() * ratio_threshold)
		{
			std::cout << "\nFound a model!\nNumber of inliers: " << inlier_indices.size() << std::endl;
			inlier_indices.insert(inlier_indices.end(),sample_indices.begin(),sample_indices.end());

			sampleFromX1X2(X1,X2,inlier_indices,sample_X1,sample_X2);


			//std::ofstream file2("file2.txt",std::ofstream::trunc);
			for(size_t k = 0; k < sample_X1.rows(); k++)
			{
				//file2 << sample_X1.row(k) << "   " << sample_X2.row(k) << std::endl;
			}

			//file2.close();

			estimateHomography(sample_X1,sample_X2,H);
			std::cout << "Original number of SIFT matches: " << X1.rows() << std::endl;
			std::cout << "Average reprojection error: " << calcAvgReprojectionError(sample_X1,sample_X2,H) << std::endl;
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
		sampleFromX1X2(X1,X2,largest_support,sample_X1,sample_X2);
		estimateHomography(sample_X1,sample_X2,H);
		inlier_indices = largest_support;
		std::cout << "Number of inliers: " << largest_support.size() << std::endl;
		std::cout << "Original number of SIFT matches: " << X1.rows() << std::endl;
		std::cout << "Average reprojection error: " << calcAvgReprojectionError(X1,X2,H) << std::endl;
	}

	else
		std::cout << "Could not find a model!" << std::endl;
}

void Panaroma::warpImage(cv::Mat src_img, const Eigen::Matrix3f &H, cv::Mat &dst_img)
{
	cv::Mat map_x, map_y;
	dst_img = cv::Mat::zeros(2*src_img.rows,2*src_img.cols,src_img.type());
	
	cv::hconcat(src_img,cv::Mat::zeros(src_img.size(),src_img.type()),src_img);
	cv::vconcat(src_img,cv::Mat::zeros(src_img.size(),src_img.type()),src_img);

	map_x.create(src_img.size(),CV_32FC1);
	map_y.create(src_img.size(),CV_32FC1);
	
	for(int i = 0; i < dst_img.rows; i++)
		for(int j = 0; j < dst_img.cols; j++)
		{
			Eigen::Vector3f x(j,i,1);
			Eigen::Vector3f px = H * x;
			px = px/px(2);

			map_x.at<float>(i,j) = px(0);
			map_y.at<float>(i,j) = px(1);
		}

	cv::remap(src_img,dst_img,map_x,map_y,CV_INTER_LINEAR);

	cv::Mat mosaic, temp;
	cv::subtract(dst_img,src_img,temp);
	cv::add(src_img,temp,mosaic);
	cv::imshow("mosaic",mosaic);
}

void Panaroma::stitch(cv::Mat img1, cv::Mat img2, cv::Mat &result)
{
	cv::hconcat(img1,cv::Mat::zeros(img1.size(),img1.type()),img1);
	cv::vconcat(img1,cv::Mat::zeros(img1.size(),img1.type()),img1);

	cv::Mat temp;
	cv::subtract(img1,img2,temp);
	cv::add(img2,temp,result);
}

void Panaroma::run(const float &dist_threshold, const float &ratio_threshold)
{
	Eigen::MatrixXf X1,X2;
    generateMatches(images[0],images[1],X1,X2);

    Eigen::MatrixXf H;
    std::vector<int> inlier_indices;
    estimateRansacHomography(X1,X2,dist_threshold,ratio_threshold,H,inlier_indices);

	//std::vector<cv::Point2f> x1,x2;
	//cv::Point2f pt;
	//Eigen::Vector3f p;
    //for(size_t i = 0; i < X1.rows(); i++)
    //{
		//p = X1.row(i);
	  	//pt.x = p(0);
		//pt.y = p(1);
		//x1.push_back(pt);
		
		//p = X2.row(i);
		//pt.x = p(0);
		//pt.y = p(1);
		//x2.push_back(pt);
    //}
    
    //cv::Mat cv_H = cv::findHomography(x1,x2,cv::RANSAC);
    //cv2eigen(cv_H, H);

	//std::cout << "OpenCV H:\n"  << H << std::endl;

    cv::Mat testImage, mosaic;
    //cv::warpPerspective(images[1],testImage,cv_H,images[0].size());

    warpImage(images[1],H,testImage);
    cv::imshow("warped",testImage);

    stitch(images[0],testImage,mosaic);
	cv::imshow("Mosaic",mosaic);
    cv::waitKey(0);
}
