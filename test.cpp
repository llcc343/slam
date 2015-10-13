#include "FeatureMatcher.h"

int main()
{
    cv::Mat rgb1=cv::imread("/home/lc/workspace/dataset2/rgb_index/173.png");
    cv::Mat rgb2=cv::imread("/home/lc/workspace/dataset2/rgb_index/173.png");

    //load pcd
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc1(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc2(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::io::loadPCDFile("/home/lc/workspace/dataset2/pcd/173.pcd",*pc1);
    pcl::io::loadPCDFile("/home/lc/workspace/dataset2/pcd/173.pcd",*pc2);

    FlannMatcher matcher;

    //matcher.getMatches(pc1,pc2,rgb1,rgb2);
    
    Eigen::Matrix4f H;
    std::vector<cv::DMatch> bestMatches;
    bool valdTrans=matcher.getFinalTransform(rgb1,rgb2,pc1,pc2,bestMatches,H);
    
    if(valdTrans)
    {
       cout<<"T is: "<<endl<<H<<endl;
       double translation;
       double angle[3];
       
       Eigen::Matrix4f H1=Eigen::Matrix4f::Identity();
       matcher.calcTranslate(H1,translation);
       matcher.calcRotation(H1,angle);
       cout<<"angle Thresh is: "<<5*M_PI/180<<endl;
       cout<<"translation is: "<<translation<<endl;
       cout<<"angle is roll: "<<angle[0]<<",pitch: "<<angle[1]<<",yaw: "<<angle[2]<<endl;
    }
    else
    {
       cout<<"there is no valid transform!!!"<<endl;
       cout<<"T is: "<<endl<<H<<endl;
    }
    
  
    cv::waitKey(0);

    return 0;
}
