#include "slam.h"

#include <iostream>

using namespace std;

int main()
{
    GraphicEnd* pGraphicEnd=new GraphicEnd();
    SLAMEnd* pSLAMEnd = new SLAMEnd();

    pSLAMEnd->init( pGraphicEnd );
    pGraphicEnd->init( pSLAMEnd );

    int loop=1;
    for(int i=1;i<1200;i++)
    {
        pGraphicEnd->run();
        //if(!pGraphicEnd->run2())
        //    break;
	loop++;
        cout<<"loop times= "<<loop<<endl;
    }
    
    //pGraphicEnd->loop_fout.close();
    cout<<"Total KeyFrame: "<<pGraphicEnd->keyframes.size()<<endl;
    pSLAMEnd->globalOptimizer.save( "/home/lc/workspace/paper_related/Appolo/test/result/final.g2o" );

    pGraphicEnd->saveFinalResult();
    
    delete pGraphicEnd;
    delete pSLAMEnd;

    return 0;
}
