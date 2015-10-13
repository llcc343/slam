#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include<algorithm>

#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>


using namespace std;

//the following are UBUNTU/LINUX ONLY terminal color codes.
#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */

class FeatureMatcher
{
 public:
    FeatureMatcher();
    ~FeatureMatcher();
    
    void getGray(cv::Mat& image, cv::Mat& gray);
    
    void finalRefine(cv::Mat& src,cv::Mat& tgt,vector<cv::DMatch>& matches,
                     vector<cv::Point2f>& src_points,vector<cv::Point2f>& tgt_points);
    
    void findMatches(cv::Mat& queryDescriptors,cv::Mat& trainDescriptors,
                     std::vector<cv::DMatch>& matches);
    
    bool refineMatchesWithHomography(std::vector<cv::KeyPoint>& queryKeypoints,
                                                 std::vector<cv::KeyPoint>& trainKeypoints, 
                                                 float reprojectionThreshold,
                                                 std::vector<cv::DMatch>& matches,
                                                 cv::Mat& homography);                

    void findMatches(const std::vector<cv::KeyPoint>& source_keypoints,
                     const cv::Mat& source_descriptors,
                     const std::vector<cv::KeyPoint>& target_keypoints,
                     const cv::Mat& target_descriptors,
                     const int image_height,
                     const int image_width,
                     std::vector<cv::DMatch >& matches);
                     
    void detectFeatures(cv::Mat& image,vector<cv::KeyPoint>& keypoints);                 

    void extractFeatrues(cv::Mat& image,vector<cv::KeyPoint>& keypoints,cv::Mat& descriptors);

    //set feature detector
    void setFeatureDetector(cv::Ptr<cv::FeatureDetector>& detect) 
    {
        detector= detect;
    }
    // Set descriptor extractor
    void setDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor>& desc) 
    {
        extractor= desc;
    }

    void useORB(bool use)
    {
        use_orb=use;
    }

 public:
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    bool use_orb;
    bool debug;
    bool enableRatioTest;
    float homographyReprojectionThreshold;
    bool enableFinalRefine;
};
