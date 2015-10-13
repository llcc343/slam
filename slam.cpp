#define ENABLE_EXTRACTOR_ORB 0
#define ENABLE_EXTRACTOR_GPU_SURF 0
#define ENABLE_EXTRACTOR_SURF 0
#define ESTIMATE_RANSAC 0
#define ESTIMATE_RANSAC_SVD 0
#define REFINE_GICP 0
#define REFINE_PCL 0

#include "slam.h"

//CV
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

//Std
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <algorithm>

//PCL

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
//G2O

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>

#if ENABLE_EXTRACTOR_ORB
#include "FeatureExtractor_ORB.h"
#elif ENABLE_EXTRACTOR_GPU_SURF
#include "FeatureExtractor_GPU_SURF.h"
#elif ENABLE_EXTRACTOR_SURF
#include "FeatureExtractor_SURF.h"
#endif

#include "FeatureMatcher.h"

#if ESTIMATE_RANSAC
#include "TransformEstimate_RANSAC.h"
#elif ESTIMATE_RANSAC_SVD
#include "TransformEstimate_RANSAC_SVD.h"
#endif

#if REFINE_GICP
#include "Refiner_GICP.h"
#elif REFINE_PCL
#include "Refiner_PCL.h"
#endif

//#include "test2/FeatureMatcher1.h"
using namespace std;
using namespace cv;
using namespace g2o;
using namespace Eigen;

//static const char *VOC_FILE = "/home/lc/workspace/paper_related/Appolo/test/surf64_k10L6.voc.gz";
typename Surf64LoopDetector::Parameters params;

GraphicEnd::GraphicEnd():
    currCloud(new PointCloud),m_height(480),m_width(640),
    m_vocPath("/home/lc/workspace/paper_related/Appolo/test/surf64_k10L6.voc.gz"),
    bow_voc(m_vocPath),db(bow_voc,false,0),//bow_detector(bow_voc,params),
    loop_fout("/home/lc/workspace/paper_related/Appolo/test/result/loop.txt")
{  
    //matcher=FlannMatcher();
    use_voxel=false;

    //bow initialize
        /***********************bow init*******************************/
    params.image_rows=m_height;
    params.image_cols=m_width;
    params.use_nss = true; // use normalized similarity score instead of raw score
    params.alpha = 0.3; // nss threshold
    params.k = 1; // a loop must be consistent with 1 previous matches
    params.geom_check = GEOM_DI; // use direct index for geometrical checking
    params.di_levels =2; // use two direct index levels
    
    //cout<<"Initiate loop detector with the vocabulary !!!"<<endl;
    
    /***********************************************************/
}

GraphicEnd::~GraphicEnd()
{
    
}

void GraphicEnd::init(SLAMEnd* pSLAMEnd)
{
    cout<<"*****************************"<<endl;
    cout<<"slam init....!!!"<<endl;
    cout<<"*****************************"<<endl;
    cout << "Loading " <<"vocabulary..." << endl;


    _pSLAMEnd=pSLAMEnd;
    index=1;
    rgbPath="/media/新加卷/dataset/dataset1/rgb_index/";
    depPath="/media/新加卷/dataset/dataset1/dep_index/";
    pcdPath="/media/新加卷/dataset/dataset1/pcd/";
    //read first image
    readimage();
    cout<<"index is: "<<index<<endl;
    
    currKF.id=0;
    currKF.frame_index=index;
    currKF.fullpose=Eigen::Isometry3d::Identity();
    currKF.frame.rgb=currRGB.clone();
    currKF.frame.dep=currDep.clone();
    //pcl::copyPointCloud(*currCloud,*(currKF.frame.cloud));
    //currKF.frame.cloud=currCloud;

    previous=currKF;
    //set first frame as keyframe
    keyframes.push_back(currKF);
    
    /******************************************************/
    //set first estimate
    double initQua[7]={1.2764 ,-0.9763, 0.6837, 0.8187, 0.3639, -0.1804, -0.4060};
    Eigen::Quaterniond q(initQua[6],initQua[3],initQua[4],initQua[5]);
    Eigen::Matrix3d r=q.toRotationMatrix();
    Eigen::AngleAxisd angle(r);
    Eigen::Isometry3d T=Eigen::Isometry3d::Identity();
    T=angle;
    
    T(0,3)=initQua[0];
    T(1,3)=initQua[1];
    T(2,3)=initQua[2];
    /*****************************************************/
    //put first image into slamEnd
    SparseOptimizer& opt=_pSLAMEnd->globalOptimizer;
    VertexSE3* v=new VertexSE3();
    v->setId(currKF.id);
    v->setEstimate(Eigen::Isometry3d::Identity());
    //v->setEstimate(T);
    v->setFixed(true);
    opt.addVertex(v);
    index++;

    /*
    //add the first image into database
    vector<cv::KeyPoint> keys;
    vector<FSurf64::TDescriptor> descrip;
    DetectionResult result;
    bow_extractor(currKF.frame.rgb,keys,descrip);

    bow_detector.detectLoop(keys,descrip,result);
 */
    
    vector<cv::KeyPoint> keys;
    vector<vector<float> > descrip;
    loadFeatures(currKF.frame.rgb,descrip,keys);
    db.add(descrip);
    cout<<"*****************************"<<endl;
}

void GraphicEnd::run()
{    
    readimage();
    cout<<"index is: "<<index<<endl;
    
    present.frame.rgb=currRGB.clone();
    present.frame.dep=currDep.clone();
    //present.frame.cloud=currCloud;
    //pcl::copyPointCloud(*currCloud,*(present.frame.cloud));
    
    Eigen::Matrix4f H;
    //bool validTrans=calcTrans(keyframes.back(),present,H,true);
    bool validTrans=calcTrans2(keyframes.back(),present,H);
    FlannMatcher matcher;
    bool isKeyframe=matcher.isKeyframe(H);
    //bool isKeyframe=isKeyframe2(H);
   
    Eigen::Isometry3d T(H.cast<double>());
    T=T.inverse();
    
    cout<<"T is:"<<endl<<T.matrix()<<endl;
    /*
    //calc present fullpose
    if(!validTrans)
    {
        cout<<"this frame is lost!!!"<<endl;
        lost++;
    }
    else
    {
        present.fullpose=previous.fullpose*T;
        FlannMatcher matcher;
        Eigen::Matrix4f H1=present.fullpose.matrix().cast<float>();
        bool isKeyframe=matcher.isKeyframe(H1);
        if(isKeyframe)
        {
            //generated keyframes
            generateKeyframe(T);
            //check the loop
            loopClosure();
            lost=0;      
        }
        else
        {
            cout<<"update robot pose!!!"<<endl;
            lost=0;
        }
        previous=present;
    }
    */
    
    if(!validTrans)
    {
        cout<<"this frame is lost!!!"<<endl;
        lost++;
    }
    else if(validTrans && isKeyframe)
    {
        //generated keyframes
        generateKeyframe(T);
        //check the loop
        //loopClosure();
        //Bowloopclosure();
        checkLoopClosure();
        lost=0;
    }
    else if(validTrans && !isKeyframe)
    {
       cout<<"update robot pose!!!"<<endl;
       lost=0;
    }
    
    if(lost>5)
    {
        cout<<"robot is lost, excuse lost recovery!!!!"<<endl;
        lostRecovery();
     
    }
  
    cout<<"current keyframe size is: "<<keyframes.size()<<endl;
    index++;
    
}

bool GraphicEnd::run2()
{
    readimage();
    cout<<"index is: "<<index<<endl;
    
    present.frame.rgb=currRGB.clone();
    present.frame.dep=currDep.clone();
    //present.frame.cloud=currCloud;
    pcl::copyPointCloud(*currCloud,*(present.frame.cloud));
    
    Eigen::Matrix4f H;
    bool validTrans=calcTrans(previous,present,H,true);
    
    Eigen::Isometry3d T(H.cast<double>());
    T=T.inverse();
    
    if(!validTrans)
    {
        cerr<<"invalid sequences!!!"<<endl;
        return false;
    }
    
    present.fullpose=previous.fullpose*T;
    
    FlannMatcher matcher;
    Eigen::Matrix4f H1=present.fullpose.matrix().cast<float>();
    bool isKeyframe=matcher.isKeyframe(H1);
    if(isKeyframe)
    {
        //generated keyframes
        generateKeyframe(T);
        //check the loop
        loopClosure();          
    }
    
    previous.frame.rgb=present.frame.rgb.clone();
    previous.frame.dep=present.frame.dep.clone();
    pcl::copyPointCloud(*(present.frame.cloud),*(previous.frame.cloud));
    
    cout<<"current keyframe size is: "<<keyframes.size()<<endl;
    index++;
    
    return true;
}


void GraphicEnd::readimage()
{
    cout<<"loading data from dataset...."<<endl;
    ss<<rgbPath<<index<<".png";
    currRGB=cv::imread(ss.str());
    ss.str("");
    ss.clear();

    ss<<depPath<<index<<".png";
    currDep=cv::imread(ss.str(),-1);
    ss.str("");
    ss.clear();

    ss<<pcdPath<<index<<".pcd";
    PointCloud::Ptr currCloudTmp(new PointCloud);
    pcl::io::loadPCDFile(ss.str(),*currCloudTmp);
    //in order to release memery stress,
    //we use filter to make pointcloud into
    //no order pointcloud(h=1)
    //downsamplePointCloud(currCloudTmp,currCloud);
    
    ss.str("");
    ss.clear();
}


void GraphicEnd::generateKeyframe(Eigen::Isometry3d& T)
{
    cout<<"generating keyframes..."<<endl;
    currKF.id++;
    currKF.frame_index=index;
    currKF.frame.rgb=present.frame.rgb.clone();
    currKF.frame.dep=present.frame.dep.clone();
    //currKF.frame.cloud=present.frame.cloud;
    pcl::copyPointCloud(*(present.frame.cloud),*(currKF.frame.cloud));
    currKF.fullpose=present.fullpose;
    //keyframes.reserve(100);

    keyframes.push_back(currKF);
    //cout<<"keyframes capacity is: "<<keyframes.capacity()<<endl;
    
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    VertexSE3* v = new VertexSE3();
    v->setId( currKF.id );
    v->setEstimate( Eigen::Isometry3d::Identity() );
    opt.addVertex( v );
    cout<<"add vertex success!!!"<<endl;
    //edge
    EdgeSE3* edge = new EdgeSE3();
    edge->vertices()[0] = opt.vertex( currKF.id - 1 );
    edge->vertices()[1] = opt.vertex( currKF.id );
    Eigen::Matrix<double, 6,6> information = Eigen::Matrix<double, 6, 6>::Identity();
    information(0, 0) = information(2,2) = 100;
    information(1, 1) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100; 
    edge->setInformation( information );
    edge->setMeasurement( T );
    opt.addEdge( edge );
    cout<<"add edge success!!!"<<endl;
}

//use featurematcher2
bool GraphicEnd::calcTrans2(KEYFRAME& frame1,KEYFRAME& frame2,Eigen::Matrix4f& H)
{
     bool result;
     FlannMatcher2 matcher;
      vector<cv::DMatch> bestMatches;
      Eigen::Matrix4f H1;
     result=matcher.getFinalTransform(frame1.frame.rgb,frame2.frame.rgb,frame1.frame.dep,
                                      frame2.frame.dep,frame1.frame.cloud,frame2.frame.cloud,
                                      bestMatches,H);
    /*
    PointCloud::Ptr aligned_cloud(new PointCloud);
    
    pcl::IterativeClosestPoint<PointT,PointT> *icp;
    icp=new pcl::GeneralizedIterativeClosestPoint<PointT,PointT>();
    //icp=new pcl::IterativeClosestPoint<PointT,PointT>();
    
    //set icp parameters
    icp->setMaximumIterations(16);
    icp->setMaxCorrespondenceDistance(0.1);
    icp->setRANSACOutlierRejectionThreshold(0.05);
    icp->setTransformationEpsilon(1e-5);
    
    icp->setInputCloud(frame1.frame.cloud);
    icp->setInputTarget(frame2.frame.cloud);
    
    
    if(result)
    {
        icp->align(*aligned_cloud,H1);
        if(icp->hasConverged())
        {
           H=icp->getFinalTransformation();
            cout<<"GICP Refine H is:"<<endl<<H<<endl;
         }
        else
        {
            cout<<"GICP has not converged!!"<<endl;
            H=H1;
        }
    }
   */
     return result;
}


//use featurematcher1
bool GraphicEnd::calcTrans(KEYFRAME& frame1,KEYFRAME& frame2,Eigen::Matrix4f& H,bool gicpRefine)
{
    bool result;
    bool validTrans;
    
    vector<cv::DMatch> bestMatches;
    FlannMatcher matcher;
    Eigen::Matrix4f H1;

    //here we use depth image instead of cloud
    validTrans=matcher.getFinalTransform(frame1.frame.rgb,frame2.frame.rgb,frame1.frame.dep,
                                         frame2.frame.dep,bestMatches,H1);
    
    //validTrans=matcher.getFinalTransform(frame1.frame.rgb,frame2.frame.rgb,frame1.frame.cloud,
    //                                          frame2.frame.cloud,bestMatches,H1);
    
    //validTrans=matcher.getFinalTransform(frame2.frame.rgb,frame1.frame.rgb,frame2.frame.cloud,
    //                                         frame1.frame.cloud,bestMatches,H);
    
    //if(!validTrans)
     //   H1=Eigen::Matrix4f::Identity();
     if(gicpRefine)
     {
    //use GICP,if it do not converge return false
    //1.first downsample the pointcloud
    //PointCloud::Ptr cloud1(new PointCloud);
    //PointCloud::Ptr cloud2(new PointCloud);
    PointCloud::Ptr aligned_cloud(new PointCloud);
    
    //downsamplePointCloud(frame1.frame.cloud,cloud1);
    //downsamplePointCloud(frame2.frame.cloud,cloud2);
    //initial gicp
    pcl::IterativeClosestPoint<PointT,PointT> *icp;
    icp=new pcl::GeneralizedIterativeClosestPoint<PointT,PointT>();
    //icp=new pcl::IterativeClosestPoint<PointT,PointT>();
    
    //set icp parameters
    icp->setMaximumIterations(16);
    icp->setMaxCorrespondenceDistance(0.1);
    icp->setRANSACOutlierRejectionThreshold(0.05);
    icp->setTransformationEpsilon(1e-5);
    
    icp->setInputCloud(/*cloud1*/frame1.frame.cloud);
    icp->setInputTarget(/*cloud2*/frame2.frame.cloud);
    //icp->align(*aligned_cloud);
    
    
    if(validTrans)
        icp->align(*aligned_cloud,H1);
    else
        icp->align(*aligned_cloud);
    
    
    H=icp->getFinalTransformation();
    if(icp->hasConverged())
    {
        //H=icp->getFinalTransformation();
        cout<<"GICP Refine H is:"<<endl<<H<<endl;
    }
    else
    {
        cout<<"GICP has not converged!!"<<endl;
    }
    result= icp->hasConverged();
    }
    else
    {
       H=H1;
       result= validTrans;
    }
    //return validTrans;
    return result;
}

bool GraphicEnd::isKeyframe2(Eigen::Matrix4f& T)
{
      bool isKey;
      Eigen::Matrix3f rotate;
      rotate=T.block<3,3>(0,0);
      
      cv::Mat R,rvec;
      cv::eigen2cv(rotate,R);
      cv::Rodrigues(R,rvec);
      
      float trans[1][3]={T(0,3),T(1,3),T(2,3)};
      cv::Mat t(1,3,CV_64F,trans);
      float norm_transformation=cv::min(cv::norm(rvec),2*M_PI-cv::norm(rvec))+cv::norm(t);
      
      if(norm_transformation>=0.2 && norm_transformation<0.8)
         isKey=true;
      else
         isKey=false;
      
      return isKey;
}


void GraphicEnd::downsamplePointCloud(PointCloud::Ptr& pc_in,PointCloud::Ptr& pc_downsampled)
{
    if(use_voxel)
    {
        pcl::VoxelGrid<pcl::PointXYZRGB> grid;
        grid.setLeafSize(0.05,0.05,0.05);
        grid.setFilterFieldName ("z");
        grid.setFilterLimits (0.0,5.0);

        grid.setInputCloud(pc_in);
        grid.filter(*pc_downsampled);
    }
    else
    {
        int downsamplingStep=8;
        static int j;j=0;
        std::vector<double> xV;
        std::vector<double> yV;
        std::vector<double> zV;
        std::vector<double> rV;
        std::vector<double> gV;
        std::vector<double> bV;

        pc_downsampled.reset(new pcl::PointCloud<pcl::PointXYZRGB> );
        pc_downsampled->points.resize(640*480/downsamplingStep*downsamplingStep);
        for(int r=0;r<480;r=r+downsamplingStep)
        {
            for(int c=0;c<640;c=c+downsamplingStep)
            {
                int nPoints=0;
                xV.resize(downsamplingStep*downsamplingStep);
                yV.resize(downsamplingStep*downsamplingStep);
                zV.resize(downsamplingStep*downsamplingStep);
                rV.resize(downsamplingStep*downsamplingStep);
                gV.resize(downsamplingStep*downsamplingStep);
                bV.resize(downsamplingStep*downsamplingStep);
                
                for(int r2=r;r2<r+downsamplingStep;r2++)
                {
                    for(int c2=c;c2<c+downsamplingStep;c2++)
                    {
                        //Check if the point has valid data
                        if(pcl_isfinite (pc_in->points[r2*640+c2].x) &&
                           pcl_isfinite (pc_in->points[r2*640+c2].y) &&
                           pcl_isfinite (pc_in->points[r2*640+c2].z) &&
                           0.3<pc_in->points[r2*640+c2].z &&
                           pc_in->points[r2*640+c2].z<5)
                        {
                            //Create a vector with the x, y and z coordinates of the square region and RGB info
                            xV[nPoints]=pc_in->points[r2*640+c2].x;
                            yV[nPoints]=pc_in->points[r2*640+c2].y;
                            zV[nPoints]=pc_in->points[r2*640+c2].z;
                            rV[nPoints]=pc_in->points[r2*640+c2].r;
                            gV[nPoints]=pc_in->points[r2*640+c2].g;
                            bV[nPoints]=pc_in->points[r2*640+c2].b;
                            
                            nPoints++;
                        }
                    }
                }
                
                if(nPoints>0)
                {
                    xV.resize(nPoints);
                    yV.resize(nPoints);
                    zV.resize(nPoints);
                    rV.resize(nPoints);
                    gV.resize(nPoints);
                    bV.resize(nPoints);
                    
                    //Compute the mean 3D point and mean RGB value
                    std::sort(xV.begin(),xV.end());
                    std::sort(yV.begin(),yV.end());
                    std::sort(zV.begin(),zV.end());
                    std::sort(rV.begin(),rV.end());
                    std::sort(gV.begin(),gV.end());
                    std::sort(bV.begin(),bV.end());
                    
                    pcl::PointXYZRGB point;
                    point.x=xV[nPoints/2];
                    point.y=yV[nPoints/2];
                    point.z=zV[nPoints/2];
                    point.r=rV[nPoints/2];
                    point.g=gV[nPoints/2];
                    point.b=bV[nPoints/2];
                    
                    //Set the mean point as the representative point of the region
                    pc_downsampled->points[j]=point;
                    j++;
                }
            }
        }
        pc_downsampled->points.resize(j);
        pc_downsampled->width=pc_downsampled->size();
        pc_downsampled->height=1;
    }
}
/*************************************************************************************/
//DBOW2
void GraphicEnd::loadFeatures(cv::Mat& image,
                              vector<vector<float> >& features,
                              vector<cv::KeyPoint>& keypoints)
{
    cv::SURF surf(400, 4, 2, false); //do not extend surf
    cv::Mat mask;
    vector<float> descriptors;

    surf(image,mask,keypoints,descriptors);
    changeStructure(descriptors,features,surf.descriptorSize());
}

void GraphicEnd::changeStructure(const vector<float> &plain,
                                 vector<vector<float> > &out,
                                 int L)
{
    out.resize(plain.size() / L);

  unsigned int j = 0;
  for(unsigned int i = 0; i < plain.size(); i += L, ++j)
  {
    out[j].resize(L);
    std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
  }
}

cv::Mat GraphicEnd::changeVecToMat(vector<vector<float> >& desp)
{
    cv::Mat despMat(desp.size(),64,CV_32F);
    for(int i=0;i<desp.size();++i)
    {
        for(int j=0;j<64;++j)
        {
            
            despMat.at<float>(j,i)=desp[ i ][ j ];
        }
    }
    return despMat;
}

bool GraphicEnd::checkFundumental(vector<cv::KeyPoint>& kp1,
                      vector<vector<float> > desp1,
                      vector<cv::KeyPoint>& kp2,
                      vector<vector<float> > desp2)
{
     bool results=false;
    vector<cv::DMatch> matches;
    //convert descriptors into Mat
    cv::Mat descriptors1=changeVecToMat(desp1);
    cv::Mat descriptors2=changeVecToMat(desp2);


    //flann match
    //cv::FlannBasedMatcher matcher;
    cv::BruteForceMatcher<cv::L2<float> > matcher;
    //std::vector< cv::DMatch > matches1;
    matcher.match( descriptors2, descriptors1, matches );

    /*
    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors1.rows; i++ )
    {
        double dist = matches1[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
    //-- PS.- radiusMatch can also be used here.
    for( int i = 0; i < descriptors1.rows; i++ )
    {
        if( matches1[i].distance < 2*min_dist )
        {
            matches.push_back( matches1[i]);
        }
    }

    
    //flann kdtree match
    cv::Mat m_indices(descriptors1.rows,2,CV_32S);
    cv::Mat m_dists(descriptors1.rows,2,CV_32S);
    cv::flann::Index flann_index(descriptors2,cv::flann::KDTreeIndexParams(4));
    flann_index.knnSearch(descriptors1,m_indices,m_dists,2,cv::flann::SearchParams(64));

    int* indices_ptr=m_indices.ptr<int>(0);
    float* dists_ptr=m_dists.ptr<float>(0);

    cv::DMatch match;
    //vector<cv::DMatch> matches;
    for (int i=0;i<m_indices.rows;++i) {
        if (dists_ptr[2*i]<0.6*dists_ptr[2*i+1]) {
            match.queryIdx=i;
            match.trainIdx=indices_ptr[ 2*i ];
            match.distance=dists_ptr[ 2*i ];
         
            matches.push_back(match);
        }
    }
    */
    if(matches.size()<12)
        return false;
    
    //calc fundamental mat
    vector<cv::Point2f> points1,points2;
    for(vector<cv::DMatch>::iterator itr=matches.begin();itr!=matches.end();++itr)
    {
        float x1=kp1[ itr->queryIdx ].pt.x;
        float y1=kp2[ itr->queryIdx ].pt.y;

        float x2=kp2[ itr->trainIdx ].pt.x;
        float y2=kp2[ itr->trainIdx ].pt.y;

        points1.push_back(cv::Point2f(x1,y1));
        points2.push_back(cv::Point2f(x2,y2));
    }
    vector<uchar> inliers(points1.size(),0);
    cv::Mat fundamental=cv::findFundamentalMat(cv::Mat(points1),
                                               cv::Mat(points2),
                                               inliers,
                                               CV_FM_RANSAC,
                                               3.0,
                                               0.99);
    int inlierNum=0;
    for(size_t i=0;i<inliers.size();++i)
    {
        if(inliers[ i ])
            inlierNum++;
    }

    if(inlierNum>=12)
        results=true;
    cout<<"num inliers: "<<inlierNum<<endl;
    return results;
}

void GraphicEnd::checkLoopClosure()
{
    cout<<"**************************************************"<<endl;
    cout<<"checking loop closure!!!"<<endl;
    cout<<"**************************************************"<<endl;
    
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    //first extract currKF desp and keys
    vector<cv::KeyPoint> keys;
    vector<vector<float> > descrip;
    loadFeatures(currKF.frame.rgb,descrip,keys);

    //add currKF into database
    db.add(descrip);

    //query, first is the image itself
    QueryResults ret;
    db.query(descrip,ret,10);//only get 3 highest score frame;

    //ransac check
    for(int j=1;j<ret.size();++j)
    {
        //extract chose image feature
        vector<cv::KeyPoint> keypoint;
        vector<vector<float> > descriptor;
        loadFeatures(keyframes[ret[ j ].Id].frame.rgb,descriptor,keypoint);
        
        bool ransacResult=checkFundumental(keys,descrip,keypoint,descriptor);

        //if pass the ransac check
        if(ransacResult)
        {
            Eigen::Matrix4f H;
            bool validTrans=calcTrans2(keyframes[ ret[ j ].Id ],currKF,H);

            loop_fout<<currKF.frame_index<<" and "<<keyframes[ret[ j ].Id].frame_index<<" : "<<validTrans<<endl;
            if(!validTrans || H==Eigen::Matrix4f::Identity())
                continue;
           
            Eigen::Isometry3d T(H.cast<double>());
            T = T.inverse();

            EdgeSE3* edge = new EdgeSE3();
            edge->vertices() [0] = opt.vertex( keyframes[ret[ j ].Id].id );
            edge->vertices() [1] = opt.vertex( currKF.id );
            Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
            information(0, 0) = information(2,2) = 100; 
            information(1,1) = 100;
            information(3,3) = information(4,4) = information(5,5) = 100; 
            edge->setInformation( information );
            edge->setMeasurement( T );
            edge->setRobustKernel( _pSLAMEnd->robustKernel );
            opt.addEdge( edge );
        }
    }
}

/*************************************************************************************/
//bow loopclosure
void GraphicEnd::Bowloopclosure()
{
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    vector<cv::KeyPoint> keys;
    vector<FSurf64::TDescriptor> descrip;
    DetectionResult result;
    bow_extractor(currKF.frame.rgb,keys,descrip);
    bow_detector.detectLoop(keys,descrip,result);

    if(result.detection())
    {
        cout<<"Loop found with image "<<result.match<<"!"<<endl;
        /******write loop into file*******/
        //ofstream loop_fout("/home/lc/workspace/paper_related/Appolo/test/result/loop.txt");
        
        /********************************/
        Eigen::Matrix4f H;
        bool validTrans=calcTrans(keyframes[ result.match ],currKF,H,true);

        loop_fout<<currKF.frame_index<<" and "<<keyframes[result.match].frame_index<<" : "<<validTrans<<endl;
        if(!validTrans)
           return;
           
        Eigen::Isometry3d T(H.cast<double>());
        //T = T.inverse();

        EdgeSE3* edge = new EdgeSE3();
        edge->vertices() [0] = opt.vertex( keyframes[result.match].id );
        edge->vertices() [1] = opt.vertex( currKF.id );
        Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
        information(0, 0) = information(2,2) = 100; 
        information(1,1) = 100;
        information(3,3) = information(4,4) = information(5,5) = 100; 
        edge->setInformation( information );
        edge->setMeasurement( T );
        edge->setRobustKernel( _pSLAMEnd->robustKernel );
        opt.addEdge( edge );
         
    }
}

void GraphicEnd::loopClosure()
{
    if (keyframes.size() >3 ) 
   {
    cout<<"Checking loop closure."<<endl;
    waitKey(10);
    vector<int> checked;
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;

    for (int i=-2; i>-5; i--)
    {
        int n = keyframes.size() + i;
        if (n>=0)
        {
            KEYFRAME& p1 = keyframes[n];
            Eigen::Matrix4f H;
            //bool validTrans=calcTrans(p1,currKF,H);
            bool validTrans=calcTrans2(p1,currKF,H);
            if(!validTrans)
               continue;
            
            Eigen::Isometry3d T(H.cast<double>());
            T = T.inverse();
            //
            EdgeSE3* edge = new EdgeSE3();
            edge->vertices() [0] = opt.vertex( keyframes[n].id );
            edge->vertices() [1] = opt.vertex( currKF.id );
            Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
            information(0, 0) = information(2,2) = 100; 
            information(1,1) = 100;
            information(3,3) = information(4,4) = information(5,5) = 100; 
            edge->setInformation( information );
            edge->setMeasurement( T );
            edge->setRobustKernel( _pSLAMEnd->robustKernel );
            opt.addEdge( edge );
        }
        else
            break;
    }
    //
    cout<<"checking seeds, seed.size()"<<seed.size()<<endl;
    vector<int> newseed;
    for (size_t i=0; i<seed.size(); i++)
    {
        KEYFRAME& p1 = keyframes[seed[i]];
        //Eigen::Isometry3d T = calcTrans( p1, currKF.frame ).T;
        Eigen::Matrix4f H;
        //bool validTrans=calcTrans(p1,currKF,H);
         bool validTrans=calcTrans2(p1,currKF,H);  
        if(!validTrans)
            continue;
            
        Eigen::Isometry3d T(H.cast<double>());  
         
        T = T.inverse();
        //
        checked.push_back( seed[i] );
        newseed.push_back( seed[i] );
        EdgeSE3* edge = new EdgeSE3();
        edge->vertices() [0] = opt.vertex( keyframes[seed[i]].id );
        edge->vertices() [1] = opt.vertex( currKF.id );
        Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
        information(0, 0) = information(2,2) = 100; 
        information(1,1) = 100;
        information(3,3) = information(4,4) = information(5,5) = 100; 
        edge->setInformation( information );
        edge->setMeasurement( T );
        edge->setRobustKernel( _pSLAMEnd->robustKernel );
        opt.addEdge( edge );
    }

    //
    cout<<"checking random frames"<<endl;
    for (int i=0; i<10; i++)
    {
        int frame = rand() % (keyframes.size() -3 ); //
        if ( find(checked.begin(), checked.end(), frame) != checked.end() ) //
            continue;
        checked.push_back( frame );
        KEYFRAME& p1 = keyframes[frame];
        //Eigen::Isometry3d T = calcTrans( p1, currKF.frame ).T;
        Eigen::Matrix4f H;
        //bool validTrans=calcTrans(p1,currKF,H);
        bool validTrans=calcTrans2(p1,currKF,H);  
          
        if(!validTrans)
             continue;
            
        Eigen::Isometry3d T(H.cast<double>());
            
        T = T.inverse();

        newseed.push_back( frame );
        //
        cout<<"find a loop closure between kf "<<currKF.id<<" and kf "<<frame<<endl;
        EdgeSE3* edge = new EdgeSE3();
        edge->vertices() [0] = opt.vertex(keyframes[frame].id );
        edge->vertices() [1] = opt.vertex( currKF.id );
        Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
        information(0, 0) = information(2,2) = 100; 
        information(1,1) = 100;
        information(3,3) = information(4,4) = information(5,5) = 100; 
        edge->setInformation( information );
        edge->setMeasurement( T );
        edge->setRobustKernel( _pSLAMEnd->robustKernel );
        opt.addEdge( edge );
    }

    waitKey(10);
    seed = newseed;
   }
}

void GraphicEnd::lostRecovery()
{
    cout<<"Lost Recovery..."<<endl;
    //
    currKF.id++;
    currKF.frame.rgb=present.frame.rgb.clone();
    currKF.frame.dep=present.frame.dep.clone();
    //currKF.frame.cloud=present.frame.cloud;
    pcl::copyPointCloud(*(present.frame.cloud),*(currKF.frame.cloud));
    currKF.frame_index = index;
    
    //waitKey(0);
    keyframes.push_back( currKF );
    
    //
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    //
    VertexSE3* v = new VertexSE3();
    v->setId( currKF.id );
    v->setEstimate( Eigen::Isometry3d::Identity() );
    opt.addVertex( v );

   /****************************************
    //check bow loop
    //first extract currKF desp and keys
    vector<cv::KeyPoint> keys;
    vector<vector<float> > descrip;
    loadFeatures(currKF.frame.rgb,descrip,keys);

    //add currKF into database
    db.add(descrip);

    //query, first is the image itself
    QueryResults ret;
    db.query(descrip,ret,10);//only get 3 highest score frame;

    //ransac check
    for(int j=1;j<ret.size();++j)
    {
        //extract chose image feature
        vector<cv::KeyPoint> keypoint;
        vector<vector<float> > descriptor;
        loadFeatures(keyframes[ret[ j ].Id].frame.rgb,descriptor,keypoint);
        
        bool ransacResult=checkFundumental(keys,descrip,keypoint,descriptor);

        //if pass the ransac check
        if(ransacResult)
        {
            Eigen::Matrix4f H;
            bool validTrans=calcTrans2(keyframes[ ret[ j ].Id ],currKF,H);

            loop_fout<<"lost frame: "<<currKF.frame_index<<" and "<<keyframes[ret[ j ].Id].frame_index<<endl;
            
            cout<<"find loop between "<<currKF.frame_index<<" and "<<keyframes[ret[ j ].Id].frame_index<<endl;
            if(!validTrans || H==Eigen::Matrix4f::Identity())
                continue;
           
            Eigen::Isometry3d T(H.cast<double>());
            T = T.inverse();

            EdgeSE3* edge = new EdgeSE3();
            edge->vertices() [0] = opt.vertex( keyframes[ret[ j ].Id].id );
            edge->vertices() [1] = opt.vertex( currKF.id );
            Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
            information(0, 0) = information(2,2) = 100; 
            information(1,1) = 100;
            information(3,3) = information(4,4) = information(5,5) = 100; 
            edge->setInformation( information );
            edge->setMeasurement( T );
            edge->setRobustKernel( _pSLAMEnd->robustKernel );
            opt.addEdge( edge );
        }
    }
    
    
    ****************************************/
    //check loop closure
    for (int i=0; i<keyframes.size()-1; i++)
    {
        KEYFRAME& p1 = keyframes[ i ];
        //Eigen::Isometry3d T = calcTrans( p1, currKF.frame ).T;
        Eigen::Matrix4f H;
        //bool validTrans=calcTrans(p1,currKF,H);
        bool validTrans=calcTrans2(p1,currKF,H);  
            
        if(!validTrans)
            continue;
            
        Eigen::Isometry3d T(H.cast<double>());
        
        T = T.inverse();
        //
        EdgeSE3* edge = new EdgeSE3();
        edge->vertices() [0] = opt.vertex( keyframes[i].id );
        edge->vertices() [1] = opt.vertex( currKF.id );
        Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
        information(0, 0) = information(1,1) = information(2,2) = 100; 
        information(3,3) = information(4,4) = information(5,5) = 100; 
        edge->setInformation( information );
        edge->setMeasurement( T );
        edge->setRobustKernel( _pSLAMEnd->robustKernel );
        opt.addEdge( edge );
    }
    
    lost = 0;
}

void GraphicEnd::saveFinalResult()
{
    cout<<"saving final result"<<endl;
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    opt.setVerbose( true );
    opt.initializeOptimization();
    opt.optimize(200);
    opt.save("/home/lc/workspace/paper_related/Appolo/test/result/final_after.g2o");

   // savepcd();
    ofstream fout("/home/lc/workspace/paper_related/Appolo/test/result/key_frame.txt");
    for(size_t i=0;i<keyframes.size();i++)
    {
	 fout<<keyframes[i].id<<" "<<keyframes[i].frame_index<<endl;
    }
    fout.close();
    //savepcd();
}

void GraphicEnd::savepcd()
{
    cout<<"save final pointcloud"<<endl;
    SLAMEnd slam;
    slam.init(NULL);
    SparseOptimizer& opt = slam.globalOptimizer;
    opt.load("/home/lc/workspace/paper_related/Appolo/test/result/final_after.g2o");
    ifstream fin("/home/lc/workspace/paper_related/Appolo/test/result/key_frame.txt");
    PointCloud::Ptr output(new PointCloud());
    PointCloud::Ptr curr( new PointCloud());
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(0.01f, 0.01f, 0.01f );
    string pclPath ="/media/新加卷/dataset/dataset1/pcd/";

    pcl::PassThrough<PointT> pass;
    pass.setFilterFieldName("z");
    double z = 5.0;
    pass.setFilterLimits(0.0, z);
    
    while( !fin.eof() )
    {
        int frame, id;
        fin>>id>>frame;
        ss<<pclPath<<frame<<".pcd";
        
        string str;
        ss>>str;
        cout<<"loading "<<str<<endl;
        ss.clear();

        pcl::io::loadPCDFile( str.c_str(), *curr );
        //cout<<"curr cloud size is: "<<curr->points.size()<<endl;
        VertexSE3* pv = dynamic_cast<VertexSE3*> (opt.vertex( id ));
        if (pv == NULL)
            break;
        Eigen::Isometry3d pos = pv->estimate();

        cout<<pos.matrix()<<endl;
        voxel.setInputCloud( curr );
        PointCloud::Ptr tmp( new PointCloud());
        voxel.filter( *tmp );
        curr.swap( tmp );
        pass.setInputCloud( curr );
        pass.filter(*tmp);
        curr->swap( *tmp );
        //cout<<"tmp: "<<tmp->points.size()<<endl;
        pcl::transformPointCloud( *curr, *tmp, pos.matrix());
        *output += *tmp;
        //cout<<"output: "<<output->points.size()<<endl;
    }
    voxel.setInputCloud( output );
    PointCloud::Ptr output_filtered( new PointCloud );
    voxel.filter( *output_filtered );
    output->swap( *output_filtered );
    //cout<<output->points.size()<<endl;
    pcl::io::savePCDFile( "/home/lc/workspace/paper_related/Appolo/test/result/result.pcd", *output);
    cout<<"final result saved."<<endl;
}
