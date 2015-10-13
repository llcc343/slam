#ifndef BOW_LOOP_DETECTION
#define BOW_LOOP_DETECTION

#include <iostream>
#include <vector>
#include <string>

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
// DLoopDetector and DBoW2
#include "DBoW2.h"
#include "DLoopDetector.h"
#include "DUtils.h"
#include "DUtilsCV.h"
#include "DVision.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
  /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<TDescriptor> &descriptors) const ;
};


/*********************************************/
//Now only use surf64,maybe orb is a good choice
/// This functor extracts SURF64 descriptors in the required format
class BowSurfExtractor: public FeatureExtractor<FSurf64::TDescriptor>
{
public:
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const;
};


#endif

