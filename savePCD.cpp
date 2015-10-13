#include "slam.h"

#include <iostream>

using namespace std;

int main()
{
    GraphicEnd* pGraphicEnd=new GraphicEnd();
    SLAMEnd* pSLAMEnd = new SLAMEnd();

    pSLAMEnd->init( pGraphicEnd );
    pGraphicEnd->init( pSLAMEnd );

    pGraphicEnd->savepcd();
    
    delete pGraphicEnd;
    delete pSLAMEnd;

    return 0;
}
