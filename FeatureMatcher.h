#ifndef FLANNMATCHER
#define FLANNMATCHER

#include <iostream>
#include <vector>
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
using namespace std;

class FlannMatcher
{
 public:
    FlannMatcher();
    ~FlannMatcher();
    
    void evaluateTransform(Eigen::Matrix4f& transform,
                                     vector<Eigen::Vector3f>& eigenPoints1,
                                     vector<Eigen::Vector3f>& eigenPoints2,
                                     double maxError,
                                     vector<int>& inliers,
                                     double& meanError,
                                     float& ratio);
                                     
    bool getFinalTransform(cv::Mat& image1,cv::Mat& iamge2,
                                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc1,
                                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc2,
                                     std::vector<cv::DMatch>& bestMatches,
                                     Eigen::Matrix4f& bestTransform);

    //pointcloud can make memery leaks,so we use depth image instead
    bool getFinalTransform(cv::Mat& image1,cv::Mat& image2,
                           cv::Mat& depth1,cv::Mat& depth2,
                           std::vector<cv::DMatch>& bestMatches,
                           Eigen::Matrix4f& bestTransform);
                                     
    void drawInliers(cv::Mat& image1,cv::Mat& image2,
                                    vector<cv::KeyPoint>& keypoints1,
                                    vector<cv::KeyPoint>& keypoints2,
                                    vector<cv::DMatch>& matches,
                                    vector<cv::DMatch>& bestMatches);
       
    IplImage* stack_imgs( IplImage* img1, IplImage* img2 ) ; 
    
    bool isKeyframe(Eigen::Matrix4f& H);
    
    void calcTranslate(Eigen::Matrix4f& H,double& translation);
    
    void calcRotation(Eigen::Matrix4f& H,double angle[3]);
                                                                    
    void getMatches(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc1,
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc2,
                    cv::Mat& rgb1,cv::Mat& rgb2,
                    std::vector<cv::DMatch>& matches,
                    vector<cv::KeyPoint>& keypoints1,
                    vector<cv::KeyPoint>& keypoints2);

    //get matches from depth image
    void getMatches(cv::Mat& depth1,cv::Mat& depth2,
                    cv::Mat& rgb1,cv::Mat& rgb2,
                    std::vector<cv::DMatch>& matches,
                    vector<cv::KeyPoint>& keypoints1,
                    vector<cv::KeyPoint>& keypoints2);

    void projectTo3D(vector<cv::KeyPoint>& keypoints,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc1,
                     vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> >& eigenPoint);

    //project to 3d use depth image
    void projectTo3D(vector<cv::KeyPoint>& keypoints,
                     cv::Mat& depth,
                     vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> >& eigenPoint);
                     
 public:
    cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> extractor;
	int maxIterations;
    double fx,fy,cx,cy;
    double camera_factor;
};
#endif
