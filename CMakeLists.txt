# Version required
CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
SET ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)


FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(PCL 1.2 REQUIRED)
FIND_PACKAGE(G2O REQUIRED)
FIND_PACKAGE(CSparse REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)

#DBoW2
set(DBOW_INCLUDE_DIR /usr/local/include/DBoW2/)
set(DLOOPDETECTOR_INCLUDE_DIR /usr/local/include/DLoopDetector/)
set(DUTILS_INCLUDE_DIR /usr/local/include/DUtils/)
set(DUTILSCV_INCLUDE_DIR /usr/local/include/DUtilsCV/)
set(DVISION_INCLUDE_DIR /usr/local/include/DVision/)

INCLUDE_DIRECTORIES(${DBOW_INCLUDE_DIR} ${DLOOPDETECTOR_INCLUDE_DIR} ${DUTILS_INCLUDE_DIR}
  ${DUTILSCV_INCLUDE_DIR} ${DVISION_INCLUDE_DIR})

#g2o
SET(G2O_ROOT "/usr/local/")
link_directories(${G2O_ROOT}/include)
include_directories(${G2O_ROOT}/lib)

INCLUDE_DIRECTORIES(${G2O_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})
LINK_LIBRARIES(${PCL_LIBRARY_DIRS})

#pairewise
ADD_EXECUTABLE(test1 FeatureMatcher.h
  FeatureMatcher.cpp
  test.cpp)
target_link_libraries(test1 ${PCL_LIBRARIES} ${OpenCV_LIBS} )

#save pcd 
#slam
ADD_EXECUTABLE(savePCD FeatureMatcher.h
  FeatureMatcher.cpp
  FeatureMatcher2.h
  FeatureMatcher2.cpp
  slam.h
  slam.cpp
  BowLoopClosure.h
  BowLoopClosure.cpp
  savePCD.cpp
  )
target_link_libraries(savePCD ${PCL_LIBRARIES} ${OpenCV_LIBS} g2o_core pthread g2o_types_slam3d 
                       g2o_solver_csparse
                       g2o_stuff g2o_csparse_extension
                       ${CSPARSE_LIBRARY}
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDBoW2.so
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDUtils.so
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDUtilsCV.so
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDVision.so
					   )
                       


#slam
ADD_EXECUTABLE(run_slam FeatureMatcher.h
  FeatureMatcher.cpp
    FeatureMatcher2.h
  FeatureMatcher2.cpp
  slam.h
  slam.cpp
  BowLoopClosure.h
  BowLoopClosure.cpp
  run_offline.cpp
  )
target_link_libraries(run_slam ${PCL_LIBRARIES} ${OpenCV_LIBS} g2o_core pthread g2o_types_slam3d 
                       g2o_solver_csparse
                       g2o_stuff g2o_csparse_extension
                       ${CSPARSE_LIBRARY}
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDBoW2.so
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDUtils.so
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDUtilsCV.so
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDVision.so
					   )
                       

#slam
ADD_EXECUTABLE(save_Traj FeatureMatcher.h
  FeatureMatcher.cpp
    FeatureMatcher2.h
  FeatureMatcher2.cpp
  slam.h
  slam.cpp
  BowLoopClosure.h
  BowLoopClosure.cpp
  saveTraj.cpp
  )
target_link_libraries(save_Traj ${PCL_LIBRARIES} ${OpenCV_LIBS} g2o_core pthread g2o_types_slam3d 
                       g2o_solver_csparse
                       g2o_stuff g2o_csparse_extension
                       ${CSPARSE_LIBRARY}
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDBoW2.so
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDUtils.so
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDUtilsCV.so
					   ${PROJECT_SOURCE_DIR}/Thirdparty/lib/libDVision.so
					   )
                       
                       
# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
ENDIF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")             
                       

