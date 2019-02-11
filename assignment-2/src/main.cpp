// Accept image paths via cli
// Load images
// Perform feature matching - in which order?
// homography estimation

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>

#include "utils.hpp"

void generateX1X2(const std::vector<cv::Point2f> &img1_pts, const std::vector<cv::Point2f> &img2_pts, const std::vector<int> &point_indices, Eigen::MatrixXf &X1, Eigen::MatrixXf &X2)
{
	X1.resize(point_indices.size(),2);
	X2.resize(point_indices.size(),2);

	size_t j = 0;
	for(size_t i : point_indices)
	{
		cv::Point2f pt = img1_pts[i];
		X1(j,0) = pt.x;
		X1(j,1) = pt.y;

		pt = img2_pts[i];
		X2(j,0) = pt.x;
		X2(j++,1) = pt.y;
	}

	X1.conservativeResize(X1.rows(),X1.cols()+1);
	X2.conservativeResize(X1.rows(),X2.cols()+1);

	X1.col(X1.cols()-1).setOnes();
	X2.col(X2.cols()-1).setOnes();
}

void estimateHomography(const Eigen::MatrixXf &X1, const Eigen::MatrixXf &X2, Eigen::MatrixXf &H)
{
	Eigen::MatrixXf A(2*X1.rows(), 9);
	Eigen::ArrayXf h(9);

	for(int i = 0, j = 0; i < A.rows(); i+=2, j++)
	{
		A.row(i) << 0, 0, 0, -1 * X2(j,2) * X1.row(j), X2(j,1) * X1.row(j);
		A.row(i+1) << X2(j,2) * X1.row(j), 0, 0, 0, -1 * X2(j,0) * X1.row(j);
	}

	std::cout << "A:\n" << A << std::endl;

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A,Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXf V;
	V = svd.matrixV();
	h = V.col(V.cols()-1);

	H.resize(3,3);
	H.row(0) = h.segment(0,3);
	H.row(1) = h.segment(3,3);
	H.row(2) = h.segment(6,3);

	std::cout << "h:\n" << h << std::endl;
	std::cout << "H:\n" << H << std::endl;
}

void estimateRansacHomography(std::vector<cv::Point2f> &img1_pts, std::vector<cv::Point2f> &img2_pts, Eigen::MatrixXf &H)
{
	std::cout << "Estimating Homography within RANSAC..." << std::endl;
	std::vector<int> largest_support;
	Eigen::MatrixXf X1, X2;

	for(size_t i = 0; i < 500; i++)
	{
		std::cout << "\n\nIteration#" << n+1 << std::endl;
		std::vector<int> sample_indices = utils::generateRandomVector(0,img1_pts.size()-1,4);

		generateX1X2(img1_pts,img2_pts,sample_indices,X1,X2);
		estimateHomography(X2,X2,H);
	}
}

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
    cv::Mat img = cv::imread(argv[i],0);
    images.push_back(img);
  }

  std::cout << images.size() << " images loaded!";

  cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
  //cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);

  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;

  detector->detectAndCompute( images[0], cv::noArray(), keypoints1, descriptors1 );
  detector->detectAndCompute( images[1], cv::noArray(), keypoints2, descriptors2 );

  //-- Step 2: Matching descriptor vectors with a FLANN based matcher
  // Since SURF is a floating-point descriptor NORM_L2 is used
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  std::vector< std::vector<cv::DMatch> > knn_matches;
  matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.7f;
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++)
  {
  	  if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
      {
      	good_matches.push_back(knn_matches[i][0]);
      }
  }
    //-- Draw matches
  cv::Mat img_matches;
  cv::drawMatches( images[0], keypoints1, images[1], keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Show detected matches
  cv::imshow("Good Matches", img_matches );

  std::vector<cv::Point2f> img1_pts, img2_pts;

  for(size_t i = 0; i < good_matches.size(); i++)
  {

	img1_pts.push_back(keypoints1[good_matches[i].queryIdx].pt);
	img2_pts.push_back(keypoints2[good_matches[i].trainIdx].pt);
  }

  cv::waitKey(0);

  Eigen::MatrixXf H;
  estimateRansacHomography(img1_pts,img2_pts,H);
  return 0;
}
