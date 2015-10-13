#include "FeatureMatcher.h"


FlannMatcher::FlannMatcher()
{
    detector=new cv::SurfFeatureDetector();
    extractor=new cv::SurfDescriptorExtractor();
    
    maxIterations=500;
    fx=525.0;
    fy=525.0;
    cx=319.5;
    cy=239.5;
    camera_factor=5000.0;
}

FlannMatcher::~FlannMatcher()
{
    
}


void FlannMatcher::evaluateTransform(Eigen::Matrix4f& transform,
                                     vector<Eigen::Vector3f>& eigenPoints1,
                                     vector<Eigen::Vector3f>& eigenPoints2,
                                     double maxError,
                                     vector<int>& inliers,
                                     double& meanError,
                                     float& ratio)
{
    inliers.clear();
    meanError=0.0;
    ratio=0.0;

    for(unsigned int i=0;i<eigenPoints1.size();i++)
    {
        Eigen::Vector4f src(eigenPoints1[ i ][ 0 ],eigenPoints1[ i ][ 1 ],
                            eigenPoints1[ i ][ 2 ],1.);
        Eigen::Vector4f tgt(eigenPoints2[ i ][ 0 ],eigenPoints2[ i ][ 1 ],
                            eigenPoints2[ i ][ 2 ],1.);

        Eigen::Vector4f diff=(transform*src)-tgt;
        double error=diff.dot(diff);
        if(error>maxError)
            continue;
        if(!(error>=0.0))
        {
            //cerr<<"error is less than 0.0!!"<<endl;
            continue;
        }
        /*
        if(std::isnan(error))
            continue;
        */
        inliers.push_back(i);
        meanError+=sqrt(error);
    }
    //cout<<"inliers size is: "<<inliers.size()<<endl;
    if(inliers.size()>0)
        meanError/=inliers.size();
    else
        meanError=-1.;
    ratio=(float)inliers.size()/eigenPoints1.size();
} 


//pointcloud can make memery leaks,so we use depth image instead
bool FlannMatcher::getFinalTransform(cv::Mat& image1,cv::Mat& image2,
                       cv::Mat& depth1,cv::Mat& depth2,
                       std::vector<cv::DMatch>& bestMatches,
                       Eigen::Matrix4f& bestTransform)
{
    vector<cv::KeyPoint> keypoints1,keypoints2;
    vector<cv::DMatch> matches;
    
    getMatches(depth1,depth2,image1,image2,matches,keypoints1,keypoints2);
    
    vector<Eigen::Vector3f> eigenPoints1,eigenPoints2;
    for(int i=0;i<matches.size();++i)
    {
        cv::Point2f p2d1;
        cv::Point2f p2d2;

        p2d1=keypoints1[matches[i].queryIdx].pt;
        p2d2=keypoints2[matches[i].trainIdx].pt;
        
        //calculate the first x,y,z
        unsigned short d1=depth1.at<unsigned short>(round(p2d1.y),round(p2d1.x));
        double z1=double(d1)/camera_factor;
        double x1=(p2d1.x-cx)*z1/fx;
        double y1=(p2d1.y-cy)*z1/fy;

        //calculate the second x,y,x
        unsigned short d2=depth2.at<unsigned short>(round(p2d2.y),round(p2d2.x));
        double z2=double(d2)/camera_factor;
        double x2=(p2d2.x-cx)*z2/fx;
        double y2=(p2d2.y-cy)*z2/fy;

        //push them into eigenPoints
        eigenPoints1.push_back(Eigen::Vector3f(x1,y1,z1));
        eigenPoints2.push_back(Eigen::Vector3f(x2,y2,z2));
    }
    
    /***********************/
    bool validTrans=false;
    pcl::TransformationFromCorrespondences tfc;
    int k=3;
    double bestError=1E10;
    float bestRatio=0.0;
    int numValidMatches=matches.size();
    
    vector<int> bestInliersIndex;
    
    bestMatches.clear();
    
    if(numValidMatches<k)
    	return false;
    
    for(int iteration=0;iteration<maxIterations;++iteration)
    {
        tfc.reset();
    	
    	for(int i=0;i<k;++i)
    	{
    	   int id_match=rand()%numValidMatches;
    	   /*
    	   Eigen::Vector3f from(pc1->at(keypoints1[matches[id_match].queryIdx].pt.x,matches[id_match].queryIdx].pt.y).x,
    	                        pc1->at(keypoints1[matches[id_match].queryIdx].pt.x,matches[id_match].queryIdx].pt.y).y,
    	                        pc1->at(keypoints1[matches[id_match].queryIdx].pt.x,matches[id_match].queryIdx].pt.y).z);
    	   Eigen::Vector3f to(pc2->at(keypoints2[matches[id_match].trainIdx].pt.x,matches[id_match].trainIdx].pt.y).x,
    	                      pc2->at(keypoints2[matches[id_match].trainIdx].pt.x,matches[id_match].trainIdx].pt.y).y,
    	                      pc2->at(keypoints2[matches[id_match].trainIdx].pt.x,matches[id_match].trainIdx].pt.y).z);                     
    	   tfc.add(from,to);
    	   */
    	   tfc.add(eigenPoints1[id_match],eigenPoints2[id_match]);
    	}
    	Eigen::Matrix4f transformation = tfc.getTransformation().matrix();
    	
    	vector<int> indexInliers;
	double maxInlierDistance = 0.05;
	double meanError;
	float ratio;
	
	evaluateTransform(transformation,
	                  eigenPoints1,eigenPoints2,
	                  maxInlierDistance*maxInlierDistance,
	                  indexInliers,
	                  meanError,
	                  ratio);
        
        if(meanError<0 || meanError >= maxInlierDistance)
                continue;
        if (meanError < bestError)
	{
	     if (ratio > bestRatio)
			bestRatio = ratio;

	     if (indexInliers.size()<10 || ratio<0.3)
			continue;	// not enough inliers found
	}
	
	tfc.reset();
	
	for(int idInlier = 0; idInlier < indexInliers.size(); idInlier++)
	{
	    int idMatch  = indexInliers[idInlier];
	    tfc.add(eigenPoints1[idInlier],eigenPoints2[idInlier]);
	}
	transformation = tfc.getTransformation().matrix();
	
	evaluateTransform(transformation,
	                  eigenPoints1,eigenPoints2,
	                  maxInlierDistance*maxInlierDistance,
	                  indexInliers,
	                  meanError,
	                  ratio);
	                  
	if (meanError < bestError)
	{
	     if (ratio > bestRatio)
			bestRatio = ratio;

	     if (indexInliers.size()<10 || ratio<0.3)
			continue;	// not enough inliers found
			
	     bestTransform=transformation;
	     bestError=meanError;
	     //cout<<"indexInliers size is: "<<indexInliers.size()<<endl;
	     bestInliersIndex=indexInliers;
	     
	}                  
    }
    
    if(bestInliersIndex.size()>0)
    {
        std::cout<<"**********************************"<<std::endl;
        std::cout<<"we get----> "<<bestInliersIndex.size()<<"/"<<eigenPoints1.size()<<" inliers!!"<<std::endl;
        std::cout<<"inliers percentage: "<<bestInliersIndex.size()*100/eigenPoints1.size()<<"% !"<<std::endl;
        std::cout<<"**********************************"<<std::endl;
        cout<<"transformation: "<<endl<<bestTransform<<endl;
    
        for(int i=0;i<bestInliersIndex.size();++i)
	{
	    //std::cout<<"inliers i is: "<<bestInliersInliers[i]<<endl;
	    bestMatches.push_back(matches[bestInliersIndex[i]]);
	}
        validTrans=true;
        
        /*
        //draw
        cv::Mat img_matches;
        cv::drawMatches(image1,keypoints1,image2,keypoints2,
                    matches,img_matches,CV_RGB(255,0,0));
        cv::drawMatches(image1,keypoints1,image2,keypoints2,
                    bestMatches,img_matches,CV_RGB(0,255,0));
        cv::imshow("ransac matches",img_matches);
        */
        drawInliers(image1,image2,keypoints1,keypoints2,matches,bestMatches);
        cv::waitKey(10);
    }
    else
    {
      cout<<"bestRatio is: "<<bestRatio<<" ,but no valid Transform founded!!"<<endl;
      validTrans=false;
    }
   return validTrans;

}

bool FlannMatcher::getFinalTransform(cv::Mat& image1,cv::Mat& image2,
                                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc1,
                                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc2,
                                     std::vector<cv::DMatch>& bestMatches,
                                     Eigen::Matrix4f& bestTransform)
{
    vector<cv::KeyPoint> keypoints1,keypoints2;
    vector<cv::DMatch> matches;
    
    getMatches(pc1,pc2,image1,image2,matches,keypoints1,keypoints2);
    
    vector<Eigen::Vector3f> eigenPoints1,eigenPoints2;
    for(int i=0;i<matches.size();++i)
    {
        cv::Point2f p2d1;
        pcl::PointXYZRGB p3d1;
        cv::Point2f p2d2;
        pcl::PointXYZRGB p3d2;
        
        p2d1=keypoints1[matches[i].queryIdx].pt;
        p2d2=keypoints2[matches[i].trainIdx].pt;
        
        p3d1=pc1->at(p2d1.x,p2d1.y);
        p3d2=pc2->at(p2d2.x,p2d2.y);
        
        eigenPoints1.push_back(Eigen::Vector3f(p3d1.x,p3d1.y,p3d1.z));
        eigenPoints2.push_back(Eigen::Vector3f(p3d2.x,p3d2.y,p3d2.z));
    }
    
    /***********************/
    bool validTrans=false;
    pcl::TransformationFromCorrespondences tfc;
    int k=3;
    double bestError=1E10;
    float bestRatio=0.0;
    int numValidMatches=matches.size();
    
    vector<int> bestInliersIndex;
    
    bestMatches.clear();
    
    if(numValidMatches<k)
    	return false;
    
    for(int iteration=0;iteration<maxIterations;++iteration)
    {
        tfc.reset();
    	
    	for(int i=0;i<k;++i)
    	{
    	   int id_match=rand()%numValidMatches;
    	   /*
    	   Eigen::Vector3f from(pc1->at(keypoints1[matches[id_match].queryIdx].pt.x,matches[id_match].queryIdx].pt.y).x,
    	                        pc1->at(keypoints1[matches[id_match].queryIdx].pt.x,matches[id_match].queryIdx].pt.y).y,
    	                        pc1->at(keypoints1[matches[id_match].queryIdx].pt.x,matches[id_match].queryIdx].pt.y).z);
    	   Eigen::Vector3f to(pc2->at(keypoints2[matches[id_match].trainIdx].pt.x,matches[id_match].trainIdx].pt.y).x,
    	                      pc2->at(keypoints2[matches[id_match].trainIdx].pt.x,matches[id_match].trainIdx].pt.y).y,
    	                      pc2->at(keypoints2[matches[id_match].trainIdx].pt.x,matches[id_match].trainIdx].pt.y).z);                     
    	   tfc.add(from,to);
    	   */
    	   tfc.add(eigenPoints1[id_match],eigenPoints2[id_match]);
    	}
    	Eigen::Matrix4f transformation = tfc.getTransformation().matrix();
    	
    	vector<int> indexInliers;
	double maxInlierDistance = 0.05;
	double meanError;
	float ratio;
	
	evaluateTransform(transformation,
	                  eigenPoints1,eigenPoints2,
	                  maxInlierDistance*maxInlierDistance,
	                  indexInliers,
	                  meanError,
	                  ratio);
        
        if(meanError<0 || meanError >= maxInlierDistance)
                continue;
        if (meanError < bestError)
	{
	     if (ratio > bestRatio)
			bestRatio = ratio;

	     if (indexInliers.size()<10 || ratio<0.3)
			continue;	// not enough inliers found
	}
	
	tfc.reset();
	
	for(int idInlier = 0; idInlier < indexInliers.size(); idInlier++)
	{
	    int idMatch  = indexInliers[idInlier];
	    tfc.add(eigenPoints1[idInlier],eigenPoints2[idInlier]);
	}
	transformation = tfc.getTransformation().matrix();
	
	evaluateTransform(transformation,
	                  eigenPoints1,eigenPoints2,
	                  maxInlierDistance*maxInlierDistance,
	                  indexInliers,
	                  meanError,
	                  ratio);
	                  
	if (meanError < bestError)
	{
	     if (ratio > bestRatio)
			bestRatio = ratio;

	     if (indexInliers.size()<10 || ratio<0.3)
			continue;	// not enough inliers found
			
	     bestTransform=transformation;
	     bestError=meanError;
	     //cout<<"indexInliers size is: "<<indexInliers.size()<<endl;
	     bestInliersIndex=indexInliers;
	     
	}                  
    }
    
    if(bestInliersIndex.size()>0)
    {
        std::cout<<"**********************************"<<std::endl;
        std::cout<<"we get----> "<<bestInliersIndex.size()<<"/"<<eigenPoints1.size()<<" inliers!!"<<std::endl;
        std::cout<<"inliers percentage: "<<bestInliersIndex.size()*100/eigenPoints1.size()<<"% !"<<std::endl;
        std::cout<<"**********************************"<<std::endl;
        cout<<"transformation: "<<endl<<bestTransform<<endl;
    
        for(int i=0;i<bestInliersIndex.size();++i)
	{
	    //std::cout<<"inliers i is: "<<bestInliersInliers[i]<<endl;
	    bestMatches.push_back(matches[bestInliersIndex[i]]);
	}
        validTrans=true;
        
        /*
        //draw
        cv::Mat img_matches;
        cv::drawMatches(image1,keypoints1,image2,keypoints2,
                    matches,img_matches,CV_RGB(255,0,0));
        cv::drawMatches(image1,keypoints1,image2,keypoints2,
                    bestMatches,img_matches,CV_RGB(0,255,0));
        cv::imshow("ransac matches",img_matches);
        */
        drawInliers(image1,image2,keypoints1,keypoints2,matches,bestMatches);
        cv::waitKey(10);
    }
    else
    {
      cout<<"bestRatio is: "<<bestRatio<<" ,but no valid Transform founded!!"<<endl;
      validTrans=false;
    }
   return validTrans;

}

bool FlannMatcher::isKeyframe(Eigen::Matrix4f& H)
{
     bool isKey;
     double translation;
     double angle[3];
     
     calcTranslate(H,translation);
     calcRotation(H,angle);
     
     double angleThresh=5*M_PI/180;
     double transThresh=0.1;
     
     if(fabs(angle[0])>angleThresh || fabs(angle[1])>angleThresh || 
        fabs(angle[2])>angleThresh || translation>transThresh)
     	    isKey=true;
     else
         isKey=false;
     
     return isKey;
}

void FlannMatcher::calcTranslate(Eigen::Matrix4f& H,double& translation)
{
    Eigen::Vector3f trans(H(0,3),H(1,3),H(2,3));
    translation=trans.norm();
}

void FlannMatcher::calcRotation(Eigen::Matrix4f& H,double angle[3])
{
    //roll
    angle[0]=atan2(H(2,1),H(2,2));
    //PTCH
    angle[1]=asin(-H(2,0));
    //yaw
    angle[2]=atan2(H(1,0),H(0,0));
}
        
//get from opensift
//get from opensift
void FlannMatcher::drawInliers(cv::Mat& image1,cv::Mat& image2,
                                    vector<cv::KeyPoint>& keypoints1,
                                    vector<cv::KeyPoint>& keypoints2,
                                    vector<cv::DMatch>& matches,
                                    vector<cv::DMatch>& bestMatches)
{
    IplImage* stacked_img=NULL;
    IplImage test1=IplImage(image1);
    IplImage test2=IplImage(image2);
    IplImage* tmp_img1=&test1;
    IplImage* tmp_img2=&test2;
    
    stacked_img=stack_imgs(tmp_img1,tmp_img2);
    //change c to mat
    cv::Mat mat_img(stacked_img,true); //deep copy
    
    for(unsigned int i=0;i<matches.size();++i)
    {
       cv::Point2f point1;
       cv::Point2f point2;
       
       point1=keypoints1[matches[i].queryIdx].pt;
       point2.x=keypoints2[matches[i].trainIdx].pt.x;
       point2.y=keypoints2[matches[i].trainIdx].pt.y+image1.rows;
       cv::line(mat_img,point1,point2,CV_RGB(255,0,0), 1, 8, 0);
    }
    
     for(unsigned int i=0;i<bestMatches.size();++i)
    {
       cv::Point2f point1;
       cv::Point2f point2;
       
       point1=keypoints1[bestMatches[i].queryIdx].pt;
       point2.x=keypoints2[bestMatches[i].trainIdx].pt.x;
       point2.y=keypoints2[bestMatches[i].trainIdx].pt.y+image1.rows;
       cv::line(mat_img,point1,point2,CV_RGB(0,255,0), 1, 8, 0);
    }
    
    cv::imshow("ransac inliers",mat_img);
}                                    
                                    
IplImage* FlannMatcher::stack_imgs( IplImage* img1, IplImage* img2 )  
{  
    IplImage* stacked = cvCreateImage( cvSize( MAX(img1->width, img2->width),  
                                        img1->height + img2->height ),  
                                        IPL_DEPTH_8U, 3 );  
  
    cvZero( stacked );  
    cvSetImageROI( stacked, cvRect( 0, 0, img1->width, img1->height ) );  
    cvAdd( img1, stacked, stacked, NULL );  
    cvSetImageROI( stacked, cvRect(0, img1->height, img2->width, img2->height) );  
    cvAdd( img2, stacked, stacked, NULL );  
    cvResetImageROI( stacked );  
  
    return stacked;  
} 

                          
void FlannMatcher::getMatches(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc1,
                              pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc2,
                              cv::Mat& rgb1,cv::Mat& rgb2,
                              std::vector<cv::DMatch>& matches,
                              vector<cv::KeyPoint>& keypoints1,
                              vector<cv::KeyPoint>& keypoints2)
{
    //vector<cv::KeyPoint> keypoints1,keypoints2;
    cv::Mat desp1,desp2;
    vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > eigenPoint1,eigenPoint2;
    
    detector->detect(rgb1,keypoints1);
    detector->detect(rgb2,keypoints2);

    projectTo3D(keypoints1,pc1,eigenPoint1);
    projectTo3D(keypoints2,pc2,eigenPoint2);

    //extract descriptors
    extractor->compute(rgb1,keypoints1,desp1);
    extractor->compute(rgb2,keypoints2,desp2);
    
    cout<<"descriptors size is: "<<desp1.rows<<" "<<desp2.rows<<endl;
    
    //flann match
    cv::Mat m_indices(desp1.rows,2,CV_32S);
    cv::Mat m_dists(desp1.rows,2,CV_32S);
    cv::flann::Index flann_index(desp2,cv::flann::KDTreeIndexParams(4));
    flann_index.knnSearch(desp1,m_indices,m_dists,2,cv::flann::SearchParams(64));

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

    cout<<"matches size is: "<<matches.size()<<endl;
    cout<<"keypoints1 size is: "<<keypoints1.size()<<endl;
    cout<<"keypoints2 size is: "<<keypoints2.size()<<endl;
    
    /*
    //draw matches
    cv::Mat img_matches;
    cv::drawMatches(rgb1,keypoints1,rgb2,keypoints2,
                    matches,img_matches);
    cv::imshow("test matches",img_matches);
    */
}

//get matches from depth image
void FlannMatcher::getMatches(cv::Mat& depth1,cv::Mat& depth2,
                              cv::Mat& rgb1,cv::Mat& rgb2,
                              std::vector<cv::DMatch>& matches,
                              vector<cv::KeyPoint>& keypoints1,
                              vector<cv::KeyPoint>& keypoints2)
{
     //vector<cv::KeyPoint> keypoints1,keypoints2;
    cv::Mat desp1,desp2;
    vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > eigenPoint1,eigenPoint2;
    
    detector->detect(rgb1,keypoints1);
    detector->detect(rgb2,keypoints2);

    projectTo3D(keypoints1,depth1,eigenPoint1);
    projectTo3D(keypoints2,depth2,eigenPoint2);

    //extract descriptors
    extractor->compute(rgb1,keypoints1,desp1);
    extractor->compute(rgb2,keypoints2,desp2);
    
    cout<<"descriptors size is: "<<desp1.rows<<" "<<desp2.rows<<endl;
    
    //flann match
    cv::Mat m_indices(desp1.rows,2,CV_32S);
    cv::Mat m_dists(desp1.rows,2,CV_32S);
    cv::flann::Index flann_index(desp2,cv::flann::KDTreeIndexParams(4));
    flann_index.knnSearch(desp1,m_indices,m_dists,2,cv::flann::SearchParams(64));

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

    cout<<"matches size is: "<<matches.size()<<endl;
    cout<<"keypoints1 size is: "<<keypoints1.size()<<endl;
    cout<<"keypoints2 size is: "<<keypoints2.size()<<endl;
    
    /*
    //draw matches
    cv::Mat img_matches;
    cv::drawMatches(rgb1,keypoints1,rgb2,keypoints2,
                    matches,img_matches);
    cv::imshow("test matches",img_matches);
    */
}


void FlannMatcher::projectTo3D(vector<cv::KeyPoint>& keypoints,
                               pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc1,
                               vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> >& eigenPoint)
{
    cv::Point2f p2d;

    for(long i=0;i<keypoints.size();++i)
    {
        p2d=keypoints[ i ].pt;
        if(p2d.x>=640 || p2d.x<=0 ||
           p2d.y>=480 || p2d.y<=0 ||
           std::isnan(p2d.x) || std::isnan(p2d.y))
        {
            keypoints.erase(keypoints.begin()+i);
            continue;
        }

        pcl::PointXYZRGB p3d=pc1->at((int)p2d.x,(int)p2d.y);

        if(isnan(p3d.x) || isnan(p3d.y) || isnan(p3d.z) || p3d.z<=0)
        {
            keypoints.erase(keypoints.begin()+i);
            continue;
        }
        eigenPoint.push_back(Eigen::Vector3f(p3d.x,p3d.y,p3d.z));
    }
    keypoints.resize(eigenPoint.size());
}


//project to 3d use depth image
void FlannMatcher::projectTo3D(vector<cv::KeyPoint>& keypoints,
                               cv::Mat& depth,
                               vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> >& eigenPoint)
{
     cv::Point2f p2d;

    for(long i=0;i<keypoints.size();)
    {
        p2d=keypoints[ i ].pt;
        if(p2d.x>=640 || p2d.x<=0 ||
           p2d.y>=480 || p2d.y<=0 ||
           std::isnan(p2d.x) || std::isnan(p2d.y))
        {
            keypoints.erase(keypoints.begin()+i);
            continue;
        }

        unsigned short d=depth.at<unsigned short>(round(p2d.y),round(p2d.x));
        double z=double(d)/camera_factor;
        double x=(p2d.x-cx)*z/fx;
        double y=(p2d.y-cy)*z/fy;

        if(std::isnan(x) || std::isnan(y) || std::isnan(z) || z<=0)
        {
            keypoints.erase(keypoints.begin()+i);
            continue;
        }
        eigenPoint.push_back(Eigen::Vector3f(x,y,z));
        ++i;
    }
    keypoints.resize(eigenPoint.size());
}
