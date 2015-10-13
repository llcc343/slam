#include "FeatureMatcher1.h"
#include "./TransformEstimate_RANSAC.h"
#include <iostream>

using namespace std;

int main()
{
    cv::Mat rgb1=cv::imread("/home/lc/workspace/dataset2/rgb_index/1.png");
    cv::Mat rgb2=cv::imread("/home/lc/workspace/dataset2/rgb_index/2.png");

    //load pcd
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc1(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc2(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::io::loadPCDFile("/home/lc/workspace/dataset2/pcd/1.pcd",*pc1);
    pcl::io::loadPCDFile("/home/lc/workspace/dataset2/pcd/2.pcd",*pc2);

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    vector<cv::DMatch> matches;
    
    FeatureMatcher matcher;
    matcher.finalRefine(rgb1,rgb2,matches,points1,points2);

    //remove the nan point in depth
    TransformEstimate_RANSAC estimator;
    
}
