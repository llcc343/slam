#include "slam.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;


int main(int argc, char** argv)
{
    SLAMEnd slam;
    slam.init( NULL );

    SparseOptimizer& opt = slam.globalOptimizer;
    opt.load("/home/lc/workspace/paper_related/Appolo/test/result/final_after.g2o");

    ifstream fin("/home/lc/workspace/paper_related/Appolo/test/result/key_frame.txt");
    ifstream asso("/media/新加卷/dataset/dataset1/associate.txt");
    /*
    //only get the first line
    ifstream init_fin("/home/lc/workspace/paper_related/Appolo/test/result/graph/groundtruth_floor.txt");
    double initQua[7];
    double timestamp;
    init_fin>>timestamp;
    for(int i=0;i<7;i++)
    {
       init_fin>>initQua[i];
    }

    //convert into quaternion
    Eigen::Quaterniond q(initQua[6],initQua[3],initQua[4],initQua[5]);
    Eigen::Matrix3d r=q.toRotationMatrix();
    Eigen::AngleAxisd angle(r);
    Eigen::Isometry3d T=Eigen::Isometry3d::Identity();
    T=angle;
    
    T(0,3)=initQua[0];
    T(1,3)=initQua[1];
    T(2,3)=initQua[2];
    */
    stringstream ss;
    ofstream fout("/home/lc/workspace/paper_related/Appolo/test/result/graph/trajectory.txt");
    
    int jump = 1;
    while (!fin.eof())
    {
        int frame, id;
        fin>>id>>frame;
        
        string timestamp;
        for(int i=0;i<frame-jump;++i)
             getline(asso,timestamp);
        asso>>timestamp;
        //位置
        VertexSE3* pv = dynamic_cast<VertexSE3*> (opt.vertex( id) );
        if (pv == NULL)
            continue;
        double data[7];
        pv->getEstimateData( data );
        
        /*************
        Eigen::Quaterniond tmp(0,0.707,-0.707,0);
        
        Eigen::Isometry3d realPos=pos;
        
        Eigen::Matrix3d rotation=realPos.rotation();
        Eigen::Vector3d translate=realPos.translation();
        
        Eigen::Quaterniond qua(rotation);
        
        data[0]=translate(0,0);
        data[1]=translate(1,0);
        data[2]=translate(2,0);
        data[3]=qua.x();
        data[4]=qua.y();
        data[5]=qua.z();
        data[6]=qua.w();
        
        /*
        //乘以初始位移
        double result[7];
        result[0]=initQua[0]+data[0];
        result[1]=initQua[1]+data[1];
        result[2]=initQua[2]+data[2];
        
        result[3]=initQua[6]*data[3]+initQua[3]*data[6]+
                  initQua[4]*data[5]-initQua[5]*data[4];
        result[4]=initQua[6]*data[4]+initQua[4]*data[6]+
                  initQua[5]*data[3]-initQua[3]*data[5];
        result[5]=initQua[6]*data[5]+initQua[5]*data[6]+
                  initQua[3]*data[4]-initQua[4]*data[3];
        result[6]=initQua[6]*data[6]-initQua[3]*data[3]-
                  initQua[4]*data[4]-initQua[5]*data[5];  */ 
        fout<<timestamp<<" ";                 
        for (int i=0; i<7; i++)
            fout<<data[i]<<" ";
        fout<<endl;

        jump = frame;
    }
    cout<<"trajectory saved."<<endl;
    fout.close();
    fin.close();
    //init_fin.close();
    asso.close();

}
