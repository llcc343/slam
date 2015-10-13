#ifndef SLAM
#define SLAM

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

//#include "RGBDFrame.h"
#include "FeatureMatcher.h"
#include "FeatureMatcher2.h"
//bow
#include "BowLoopClosure.h"

using namespace g2o;
using namespace cv;
using namespace std;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

struct RGBDFrame
{
   RGBDFrame():cloud(new PointCloud){}
   cv::Mat rgb;
   cv::Mat dep;
   PointCloud::Ptr cloud;
};

struct KEYFRAME
{
    RGBDFrame frame;
    Eigen::Transform<double,3,Eigen::Isometry,Eigen::DontAlign> fullpose;
    int id;
    int frame_index;
};

struct FINAL_RESULT
{
    FINAL_RESULT()
    {
       T=Eigen::Isometry3d::Identity();
       numInliers=0;
       norm=0.;
    }
    Eigen::Transform<double,3,Eigen::Isometry,Eigen::DontAlign> T;
    int numInliers;
    float norm;
};

class GraphicEnd;
class SLAMEnd;

class GraphicEnd
{
 public:
    GraphicEnd();
    ~GraphicEnd();
    
    void init(SLAMEnd* pSLAMEnd);
    void run();
    bool run2();
    void readimage();
    void generateKeyframe(Eigen::Isometry3d& T);
    bool calcTrans(KEYFRAME& frame1,KEYFRAME& frame2,Eigen::Matrix4f& H,bool gicpRefine=false);
    bool calcTrans2(KEYFRAME& frame1,KEYFRAME& frame2,Eigen::Matrix4f& H);
    bool isKeyframe2(Eigen::Matrix4f& T);
    void downsamplePointCloud(PointCloud::Ptr& pc_in,PointCloud::Ptr& pc_downsampled);
    void loopClosure();
    void lostRecovery();
    void saveFinalResult();
    void savepcd();

    void Bowloopclosure();

    void loadFeatures(cv::Mat& image,vector<vector<float> >& features,vector<cv::KeyPoint>& keypoints);

    void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
                         int L);

    cv::Mat changeVecToMat(vector<vector<float> >& desp);

    bool checkFundumental(vector<cv::KeyPoint>& kp1,vector<vector<float> > desp1,vector<cv::KeyPoint>& kp2,vector<vector<float> > desp2);

    void checkLoopClosure();
    
 public:
    SLAMEnd* _pSLAMEnd;
    //FlannMatcher matcher;
    vector<KEYFRAME> keyframes;
    KEYFRAME currKF;
    KEYFRAME present;
    KEYFRAME previous;
    Mat currRGB;
    Mat currDep;
    PointCloud::Ptr currCloud;

    bool use_voxel;
    int lost;
    int index;
    string rgbPath,depPath,pcdPath;
    stringstream ss;

    vector<int> seed;

    //use for bow loop detector
    int m_height,m_width;
    string m_vocPath;
    
    BowSurfExtractor bow_extractor;
    Surf64Vocabulary bow_voc;
    Surf64LoopDetector bow_detector;
    ofstream loop_fout;

    Surf64Database db;

};

/************************************slam end************************************/
typedef BlockSolver_6_3 SlamBlockSolver;
typedef LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

class SLAMEnd
{
 public:
    SLAMEnd() {
        
    }
    ~SLAMEnd() {
        
    }
    void init(GraphicEnd* p)
    {
        _pGraphicEnd=p;
        SlamLinearSolver* linearSolver=new SlamLinearSolver();
        linearSolver->setBlockOrdering(false);
        SlamBlockSolver* blockSolver=new SlamBlockSolver(linearSolver);

        solver=new OptimizationAlgorithmLevenberg(blockSolver);
        robustKernel = RobustKernelFactory::instance()->construct( "Cauchy" );
        globalOptimizer.setVerbose( false );
        globalOptimizer.setAlgorithm( solver );
    }
 public:
    GraphicEnd* _pGraphicEnd;
    SparseOptimizer globalOptimizer;
    OptimizationAlgorithmLevenberg* solver;
    RobustKernel* robustKernel;
};

#endif
