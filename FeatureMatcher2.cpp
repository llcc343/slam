#define USE_GICP_REFINE 0

#include "FeatureMatcher2.h"

FlannMatcher2::FlannMatcher2()
{
    min_matches=16;
    max_dist_for_inliers=0.03;
    //detector=new cv::SurfFeatureDetector();
    //extractor=new cv::SurfDescriptorExtractor();
    
    detector=new cv::SiftFeatureDetector();
    extractor=new cv::SiftDescriptorExtractor();
    
    fx=525.0;
    fy=525.0;
    cx=319.5;
    cy=239.5;
    camera_factor=5000.0;
    /*
    fx=688.97646555748;
    fy=689.247554861243;
    cx=312.598620781505;
    cy=240.674170670586;
    camera_factor=1000.0;
    */
}

FlannMatcher2::~FlannMatcher2()
{
    
}



void FlannMatcher2::projectTo3D(vector<cv::KeyPoint>& keypoints,
                                cv::Mat& depth,
                                vector<Eigen::Vector3f >& eigenPoint)
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
        ++i; //must put i in the last
    }
    //keypoints.resize(eigenPoint.size());
    assert(keypoints.size()==eigenPoint.size());
}

void FlannMatcher2::getMatches(cv::Mat& depth1,cv::Mat& depth2,
                               cv::Mat& rgb1,cv::Mat& rgb2,
                               std::vector<cv::DMatch>& matches,
                               vector<cv::KeyPoint>& keypoints1,
                               vector<cv::KeyPoint>& keypoints2,
                               vector<Eigen::Vector3f >& eigenPoints1,
                               vector<Eigen::Vector3f >& eigenPoints2)
{
    cv::Mat desp1,desp2;
    
    detector->detect(rgb1,keypoints1);
    detector->detect(rgb2,keypoints2);

    projectTo3D(keypoints1,depth1,eigenPoints1);
    projectTo3D(keypoints2,depth2,eigenPoints2);

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
}

template<class InputIterator>
Eigen::Matrix4f FlannMatcher2::getTransformFromMatches(vector<Eigen::Vector3f>& eigenPoints1,
                                                       vector<Eigen::Vector3f>& eigenPoints2,
                                                       InputIterator itr_begin,
                                                       InputIterator itr_end,
                                                       bool& valid,
                                                       float max_dist_m)
{
     pcl::TransformationFromCorrespondences tfc;
     valid = true;
     vector<Eigen::Vector3f> t, f;

     for(;itr_begin!=itr_end;++itr_begin)
     {
        unsigned int this_id=itr_begin->queryIdx;
        unsigned int earlier_id=itr_begin->trainIdx;

         
         Eigen::Vector3f from(eigenPoints1[ this_id ][ 0 ],
                              eigenPoints1[ this_id ][ 1 ],
                              eigenPoints1[ this_id ][ 2 ]);
         Eigen::Vector3f to(eigenPoints2[ earlier_id ][ 0 ],
                            eigenPoints2[ earlier_id ][ 1 ],
                            eigenPoints2[ earlier_id ][ 2 ]);
         
        // Eigen::Vector3f from(eigenPoints1[this_id]);
        // Eigen::Vector3f to(eigenPoints2[earlier_id]);
         
         if(max_dist_m>0)
         {
             f.push_back(from);
             t.push_back(to);
         }
         tfc.add(from,to,1.0/to(0));
         //tfc.add(from,to);
     }

     // find smalles distance between a point and its neighbour in the same cloud
     if(max_dist_m>0)
     {
         Eigen::Matrix4f foo;

         valid=true;
         for(uint i=0;i<f.size();++i)
         {
             float d_f = (f.at((i+1)%f.size())-f.at(i)).norm();
             float d_t = (t.at((i+1)%t.size())-t.at(i)).norm();
             
             if ( abs(d_f-d_t) > max_dist_m )
             {
                 valid = false;
                 return Eigen::Matrix4f();
             }
         }
     }

     return tfc.getTransformation().matrix();
}

void FlannMatcher2::computeInliersAndError(vector<cv::DMatch>& matches,
                            Eigen::Matrix4f& transformation,
                            vector<Eigen::Vector3f>& eigenPoints1,
                            vector<Eigen::Vector3f>& eigenPoints2,
                            vector<cv::DMatch>& inliers, //output
                            double& mean_error,
                            vector<double>& errors,
                            double squaredMaxInlierDistInM)
{
    inliers.clear();
    errors.clear();

    vector<pair<float,int> > dists;
    std::vector<cv::DMatch> inliers_temp;
    
    assert(matches.size() > 0);
    mean_error = 0.0;

    for(unsigned int j=0;j<matches.size();++j)
    {
        unsigned int this_id=matches[ j ].queryIdx;
        unsigned int earlier_id=matches[ j ].trainIdx;

        Eigen::Vector4f origins(eigenPoints1[ this_id ][ 0 ],
                                eigenPoints1[ this_id ][ 1 ],
                                eigenPoints1[ this_id ][ 2 ],
                                1.);
        Eigen::Vector4f earlier(eigenPoints2[ earlier_id ][ 0 ],
                                eigenPoints2[ earlier_id ][ 1 ],
                                eigenPoints2[ earlier_id ][ 2 ],
                                1.);

        Eigen::Vector4f vec=(transformation*origins)-earlier;

        double error=vec.dot(vec);

        if(error>squaredMaxInlierDistInM)
            continue;
        if(!(error>=0.0))
            throw runtime_error("Invalid error!!!!");
        
        error=sqrt(error);
        dists.push_back(pair<float,int>(error,j));
        inliers_temp.push_back(matches[ j ]);

        mean_error+=error;
        errors.push_back(error);
    }

    if(inliers_temp.size()<3)
    {
        //cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
        //cout<<"No inliers at all!!"<<endl;
        mean_error=1e9;
    }
    else
    {
        mean_error/=inliers_temp.size();
        // sort inlier ascending according to their error
        sort(dists.begin(),dists.end());

        inliers.resize(inliers_temp.size());
        for(unsigned int i=0;i<inliers_temp.size();i++)
        {
            inliers[ i ]=matches[ dists[ i ].second ];
        }
    }

    if(!(mean_error>0))
        throw runtime_error("invalid mean error in computeInlierAndErorr!!!");
}


bool FlannMatcher2::getRelativeTransformations(vector<Eigen::Vector3f>& eigenPoints1,
                                               vector<Eigen::Vector3f>& eigenPoints2,
                                               vector<cv::DMatch>& initial_matches,
                                               Eigen::Matrix4f& result_transform,
                                               float& rmse,
                                               vector<cv::DMatch>& matches,
                                               unsigned int ransac_iterations)
{
    matches.clear();

    if(initial_matches.size()<min_matches)
    {
        cout<<"matches between this two frames is too small!!"<<endl;
        return false;
    }

    //min inliers threshold at least bigger than
    //initial matches's 20%
    unsigned int min_inlier_threshold=int(initial_matches.size()*0.2);
    vector<cv::DMatch> inlier; //hold matches support the transformation
    double inlier_error; //squre error mean

    srand((long)std::clock());

    float max_dist_m=max_dist_for_inliers;
    vector<double> dummy; //error vector

    // best values of all iterations (including invalids)
    double best_error = 1e6, best_error_invalid = 1e6;
    unsigned int best_inlier_invalid = 0, best_inlier_cnt = 0/*count best inlier number*/, valid_iterations = 0;

    Eigen::Matrix4f transformation;
  
    const unsigned int sample_size = 3;// chose this many randomly from the correspondences:

    for(unsigned int n_iter=0;n_iter<ransac_iterations;++n_iter)
    {
        std::set<cv::DMatch> sample_matches;
        std::vector<cv::DMatch> sample_matches_vector;
        while(sample_matches.size() < sample_size)
        {
            int id = rand() % initial_matches.size();
            sample_matches.insert(initial_matches.at(id));
            sample_matches_vector.push_back(initial_matches.at(id));
        }

        
        bool valid;
         transformation=getTransformFromMatches(eigenPoints1,eigenPoints2,
                                               sample_matches.begin(),
                                               sample_matches.end(),
                                               valid,
                                               max_dist_m);

        if(!valid) continue;
        if(transformation!=transformation) continue; //contain NaN

        //test whether samples are inliers (more strict than before)
        computeInliersAndError(sample_matches_vector,transformation,
                               eigenPoints1,eigenPoints2,inlier,inlier_error,
                               dummy,max_dist_m*max_dist_m);
        // cout<<" sample mean error: "<<inlier_error<<endl;
        if(inlier_error>1000) continue;

        computeInliersAndError(initial_matches,transformation,
                               eigenPoints1,eigenPoints2,inlier,inlier_error,
                               dummy,max_dist_m*max_dist_m);
        
        
        if(inlier.size()>best_inlier_invalid)
        {
            best_inlier_invalid=inlier.size();
            best_error_invalid=inlier_error;
        }

        if(inlier.size()<min_inlier_threshold || inlier_error>max_dist_m)
            continue;

        valid_iterations++;

        assert(inlier_error>0);

        //Performance hacks:
        ///Iterations with more than half of the initial_matches inlying, count twice
        if (inlier.size() > initial_matches.size()*0.5) n_iter++;
        ///Iterations with more than 80% of the initial_matches inlying, count threefold
        if (inlier.size() > initial_matches.size()*0.8) n_iter++;

        if(inlier_error<best_error)
        {
            result_transform=transformation;
            matches=inlier;
            assert(matches.size()>= min_inlier_threshold);
            best_inlier_cnt=inlier.size();
            best_error=inlier_error;
        }
        else
        {
            
        }

        double new_inlier_error;

        transformation=getTransformFromMatches(eigenPoints1,eigenPoints2,
                                               matches.begin(),
                                               matches.end(),
                                               valid);
        if(transformation!=transformation)
            continue;

        computeInliersAndError(initial_matches,transformation,
                               eigenPoints1,eigenPoints2,inlier,new_inlier_error,
                               dummy,max_dist_m*max_dist_m);

        if(inlier.size()>best_inlier_invalid)
        {
            best_inlier_invalid=inlier.size();
            best_error_invalid=inlier_error;
        }

        if(inlier.size()<min_inlier_threshold || new_inlier_error>max_dist_m)
            continue;

        assert(new_inlier_error>0);

        if(new_inlier_error<best_error)
        {
            result_transform=transformation;
            matches=inlier;
            assert(matches.size()>= min_inlier_threshold);
            rmse=new_inlier_error;
            best_error=new_inlier_error;
        }
        else
        {
            
        }
    }

    return matches.size()>=min_inlier_threshold;
}


bool FlannMatcher2::getFinalTransform(cv::Mat& image1,cv::Mat& image2,
                                      cv::Mat& depth1,cv::Mat& depth2,
                                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc1,//Must be downsampled
                                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc2,
                                      vector<cv::DMatch>& bestMatches,
                                      Eigen::Matrix4f& bestTransform)
{
    vector<cv::KeyPoint> keypoints1,keypoints2;
    vector<cv::DMatch> initial_matches;
    vector<Eigen::Vector3f> eigenPoints1,eigenPoints2;
    
    getMatches(depth1,depth2,image1,image2,
               initial_matches,
               keypoints1,keypoints2,
               eigenPoints1,eigenPoints2);
               
    const unsigned int min_matche = 16;
    if(initial_matches.size()<min_matche)
    {
        cout<<"initial matches is too small!!!"<<endl;
        return false;
    }

    float rmse=0.;
    Eigen::Matrix4f initialTrans;
    bool validTrans=getRelativeTransformations(eigenPoints1,
                                               eigenPoints2,
                                               initial_matches,
                                               initialTrans,
                                               rmse,
                                               bestMatches);
    if(!validTrans)
    {
        cout<<"found no valid transfomation!!!"<<endl;
    }
    else
    {
       //gicp refine
       #if USE_GICP_REFINE
       //first transform pc1 to p2' using ransac transform
       pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcTmp(new pcl::PointCloud<pcl::PointXYZRGB>);
       
       //pcl::transformPointCloud(*pc1,*pcTmp,initialTrans);
       
       //
       pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
       pcl::IterativeClosestPoint<pcl::PointXYZRGB,pcl::PointXYZRGB> *icp;
       icp=new pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZRGB,pcl::PointXYZRGB>();
       
       icp->setMaximumIterations(16);
       icp->setMaxCorrespondenceDistance(0.1);
       icp->setRANSACOutlierRejectionThreshold(0.05);
       icp->setTransformationEpsilon(1e-5);
       
       icp->setInputCloud(pc1);
       icp->setInputTarget(pc2);
       
       Eigen::Matrix4f icpH;
       icp->align(*aligned_cloud,bestTransform);
       
       icpH=icp->getFinalTransformation();
       
       if(!icp->hasConverged())
       {
           cout<<"GICP has not converged!!"<<endl;
           validTrans=icp->hasConverged();
       }
       else
       {
           bestTransform=icpH;
       	   //check if icp improve aligment
       	   vector<double> errors;
           double error; //mean error
           std::vector<cv::DMatch> inliers;
       	   computeInliersAndError(bestMatches,bestTransform,eigenPoints1,eigenPoints2,
       	                          inliers,error,errors,0.04*0.04);
       	   if(error>rmse+0.02)
       	   {
       	      bestTransform=initialTrans;
       	      cout<<"icp error is too large!!!"<<endl;
       	   }
       }
       #else
          bestTransform=initialTrans; //only use the ransac
       #endif
    }
    //draw
     drawInliers(image1,image2,keypoints1,keypoints2,initial_matches,bestMatches);
     cv::waitKey(5);
    return validTrans;
}

//draw
void FlannMatcher2::drawInliers(cv::Mat& image1,cv::Mat& image2,
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

IplImage* FlannMatcher2::stack_imgs( IplImage* img1, IplImage* img2 )  
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
