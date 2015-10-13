#ifndef FLANNMATCHER2
#define FLANNMATCHER2

#include <iostream>
#include <vector>
#include <set>
#include <math.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>

using namespace std;

class FlannMatcher2
{
 public:
    FlannMatcher2();
    ~FlannMatcher2();

    void projectTo3D(vector<cv::KeyPoint>& keypoints,
                     cv::Mat& depth,
                     vector<Eigen::Vector3f >& eigenPoint);
        
    void getMatches(cv::Mat& depth1,cv::Mat& depth2,
                    cv::Mat& rgb1,cv::Mat& rgb2,
                    std::vector<cv::DMatch>& matches,
                    vector<cv::KeyPoint>& keypoints1,
                    vector<cv::KeyPoint>& keypoints2,
                    vector<Eigen::Vector3f>& eigenPoints1,
                    vector<Eigen::Vector3f>& eigenPoints2);

    template<class InputIterator>
    Eigen::Matrix4f getTransformFromMatches(vector<Eigen::Vector3f>& eigenPoints1,
                                            vector<Eigen::Vector3f>& eigenPoints2,
                                            InputIterator itr_begin,
                                            InputIterator itr_end,
                                            bool& valid,
                                            float max_dist_m=-1);

    void computeInliersAndError(vector<cv::DMatch>& matches,
                                Eigen::Matrix4f& transformation,
                                vector<Eigen::Vector3f>& eigenPoints1,
                                vector<Eigen::Vector3f>& eigenPoints2,
                                vector<cv::DMatch>& inliers, //output
                                double& mean_error,
                                vector<double>& errors,
                                double squaredMaxInlierDistInM=0.0009);

    bool getRelativeTransformations(vector<Eigen::Vector3f>& eigenPoints1,
                                    vector<Eigen::Vector3f>& eigenPoints2,
                                    vector<cv::DMatch>& initial_matches,
                                    Eigen::Matrix4f& result_transform,
                                    float& rmse,
                                    vector<cv::DMatch>& matches,
                                    unsigned int ransac_iterations=1000);

    bool getFinalTransform(cv::Mat& image1,cv::Mat& image2,
                           cv::Mat& depth1,cv::Mat& depth2,
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc1,//Must be downsampled
                           pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc2,
                           vector<cv::DMatch>& bestMatches,
                           Eigen::Matrix4f& bestTransform);

    //draw functions
    void drawInliers(cv::Mat& image1,cv::Mat& image2,
                     vector<cv::KeyPoint>& keypoints1,
                     vector<cv::KeyPoint>& keypoints2,
                     vector<cv::DMatch>& matches,
                     vector<cv::DMatch>& bestMatches);

    IplImage* stack_imgs( IplImage* img1, IplImage* img2 );
    
 private:
    int min_matches;
    float max_dist_for_inliers;

    cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> extractor;
    double fx,fy,cx,cy;
    double camera_factor;
};
#endif
