#include "FeatureMatcher.h"


FeatureMatcher::FeatureMatcher()
{
    use_orb=true;
    debug=true;
    enableRatioTest=true;
    homographyReprojectionThreshold=3;
   

    if(use_orb)
    {
        //detector=new cv::OrbFeatureDetector();
        //extractor=new cv::OrbDescriptorExtractor();
        detector=new cv::ORB(1000);
        extractor=new cv::FREAK(false,false);
        //matcher=cv::DescriptorMatcher::create("BruteForce-Hamming");
        matcher=new cv::BFMatcher();
    }
    else
    {
         //using surf as defalut
        detector= new cv::SurfFeatureDetector();
        extractor= new cv::SurfDescriptorExtractor();

        //matcher=cv::DescriptorMatcher::create("FlannBased");
        matcher=new cv::FlannBasedMatcher();
    }
        
}

FeatureMatcher::~FeatureMatcher()
{
    
}



void FeatureMatcher::getGray(cv::Mat& image, cv::Mat& gray)
{
    if (image.channels()  == 3)
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, CV_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;
}

void FeatureMatcher::detectFeatures(cv::Mat& image,vector<cv::KeyPoint>& keypoints)
{
    detector->detect(image,keypoints);
    cout<<"detect "<<keypoints.size()<<" keypoints!"<<endl;
}

void FeatureMatcher::extractFeatrues(cv::Mat& image,vector<cv::KeyPoint>& keypoints,cv::Mat& descriptors)
{
    extractor->compute(image,keypoints,descriptors);
    cout<<"extracted "<<descriptors.rows<<" descriptors"<<endl;
}


//get the refined and point2f
void FeatureMatcher::finalRefine(cv::Mat& src,cv::Mat& tgt,vector<cv::DMatch>& matches,
                                 vector<cv::Point2f>& src_points,vector<cv::Point2f>& tgt_points)
{
    cv::Mat srcGray,tgtGray;
    getGray(src,srcGray);
    getGray(tgt,tgtGray);

    //extract
    vector<cv::KeyPoint> src_keypoints,tgt_keypoints;
    cv::Mat src_descriptors,tgt_descriptors;
    detectFeatures(srcGray,src_keypoints);
    extractFeatrues(srcGray,src_keypoints,src_descriptors);
    detectFeatures(tgtGray,tgt_keypoints);
    extractFeatrues(tgtGray,tgt_keypoints,tgt_descriptors);

    //get matches
    //vector<cv::DMatch> matches;
    findMatches(src_descriptors,tgt_descriptors,matches);

    //debug
    if(debug)
    {
        cv::Mat img_matches;
        cv::drawMatches(src,src_keypoints,tgt,tgt_keypoints,matches,img_matches);
        cv::imshow("knn-matching",img_matches);
        cv::waitKey(10);
    }

    //rough homography
    cv::Mat m_roughHomography;
    bool homographyFound=refineMatchesWithHomography(src_keypoints,tgt_keypoints,
                                                     homographyReprojectionThreshold,
                                                     matches,
                                                     m_roughHomography); //storw homography matrix
    if(debug)
    {
        cv::Mat img_matches;
        cv::drawMatches(src,src_keypoints,tgt,tgt_keypoints,matches,img_matches);
        cv::imshow("homgraphy matching",img_matches);
        cv::waitKey(10);
    }

    cout<<"matches size is: "<<matches.size()<<endl;
    
    //get the point2f 
    for(size_t i=0;i<matches.size();++i)
    {
        src_points.push_back(src_keypoints[matches[i].queryIdx].pt);
        tgt_points.push_back(tgt_keypoints[matches[i].trainIdx].pt);
    
    }
}

void FeatureMatcher::findMatches(cv::Mat& queryDescriptors,cv::Mat& trainDescriptors,
                                 std::vector<cv::DMatch>& matches)
{
    matches.clear();

    if (enableRatioTest)
    {
        // To avoid NaN's when best match has zero distance we will use inversed ratio. 
        const float minRatio = 1.f / 1.5f;
        
        // KNN match will return 2 nearest matches for each query descriptor
        std::vector< std::vector<cv::DMatch> > m_knnMatches;
        matcher->knnMatch(queryDescriptors,trainDescriptors, m_knnMatches, 2);

        for (size_t i=0; i<m_knnMatches.size(); i++)
        {
            const cv::DMatch& bestMatch   = m_knnMatches[i][0];
            const cv::DMatch& betterMatch = m_knnMatches[i][1];

            float distanceRatio = bestMatch.distance / betterMatch.distance;
            
            // Pass only matches where distance ratio between 
            // nearest matches is greater than 1.5 (distinct criteria)
            if (distanceRatio < minRatio)
            {
                matches.push_back(bestMatch);
            }
        }
    }
    else
    {
        // Perform regular match
        matcher->match(queryDescriptors, trainDescriptors,matches);
    }
}

bool FeatureMatcher::refineMatchesWithHomography(std::vector<cv::KeyPoint>& queryKeypoints,
                                                 std::vector<cv::KeyPoint>& trainKeypoints, 
                                                 float reprojectionThreshold,
                                                 std::vector<cv::DMatch>& matches,
                                                 cv::Mat& homography)
{
    const int minNumberMatchesAllowed = 8;

    if (matches.size() < minNumberMatchesAllowed)
        return false;
    
    // Prepare data for cv::findHomography
    std::vector<cv::Point2f> srcPoints(matches.size());
    std::vector<cv::Point2f> dstPoints(matches.size());
    
    for (size_t i = 0; i < matches.size(); i++)
    {
        srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
        dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
    }
    
    // Find homography matrix and get inliers mask
    std::vector<unsigned char> inliersMask(srcPoints.size());
    homography = cv::findHomography(srcPoints, 
                                    dstPoints, 
                                    CV_FM_RANSAC, 
                                    reprojectionThreshold, 
                                    inliersMask);
                                                                 
    
    std::vector<cv::DMatch> inliers;
    for (size_t i=0; i<inliersMask.size(); i++)
    {
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }
    
    matches.swap(inliers);
    return matches.size() > minNumberMatchesAllowed;
}

void FeatureMatcher::findMatches(const std::vector<cv::KeyPoint>& source_keypoints,
                                 const cv::Mat& source_descriptors,
                                 const std::vector<cv::KeyPoint>& target_keypoints,
                                 const cv::Mat& target_descriptors,
                                 const int image_height,
                                 const int image_width,
                                 std::vector<cv::DMatch >& matches)
{
    
}
