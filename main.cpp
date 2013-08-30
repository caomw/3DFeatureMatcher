#include <iostream>
#include <locale>
#include <fstream>
#include <iomanip>
#include <vector>
#include <math.h>

#include <lmmin.h>

#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "DescriptorsMatcher/descriptorsmatcher.h"
#include "Triangulator/singlecameratriangulator.h"
#include "Triangulator/normaloptimizer.h"
#include "Triangulator/neighborhoodsgenerator.h"
#include "pclvisualizerthread.h"
#include "tools.h"

#define IMG_1 "/home/mpp/WorkspaceTesi/loop_dataset/Images/img_0000000750.pgm"
#define IMG_2 "/home/mpp/WorkspaceTesi/loop_dataset/Images/img_0000000770.pgm"

// IMG_1 pose: TIME : 25027785 POS : 4.467813 3.420069 0.806258 0.074931 -0.160281 0.563678
// IMG_2 pose: TIME : 25694439 POS : 5.034858 3.667427 0.833424 0.014587 -0.248119 0.523502

/** Displays the usage message
 */
int help(void)
{
    std::cout << "Usage: 3dfeaturematcher -s <settings.yml>" << std::endl;
    return 0;
}

int main(int argc, char **argv) {
    
    std::cout << "Hello!" << std::endl;
    
    std::cout << std::fixed << std::setprecision(12);
    setlocale(LC_NUMERIC, "C");
    
    /////////////////////////////    
    /// Check arguments
    if (argc != 3)
    {
        help();
        exit(-1);
    }
    
    if (std::string(argv[1]) != "-s")
    {
        help();
        exit(-1);
    }
    
    /////////////////////////////    
    /// Open input files
    std::string 
        settFileName = argv[2];
    
    cv::FileStorage 
    fs;
    
    fs.open(settFileName, cv::FileStorage::READ);
    
    if (!fs.isOpened()) 
    {
        std::cerr << "Could not open settings file: " << settFileName << std::endl;
        exit(-1);
    }
    
    /////////////////////////////    
    // Get images
    cv::Mat
        img1 = cv::imread(IMG_1, CV_LOAD_IMAGE_GRAYSCALE),
        img2 = cv::imread(IMG_2, CV_LOAD_IMAGE_GRAYSCALE);
        
    /////////////////////////////    
    // Estraggo le feature e le confronto
    std::vector< cv::KeyPoint >
        kpts1, kpts2;
    cv::Mat
        desc1, desc2;
    std::vector< cv::DMatch > 
        matches;
    DescriptorsMatcher 
        dm(fs, img1, img2);
    
    dm.compareWithNNDR(fs["NNDR"]["epsilon"], matches, kpts1, kpts2, desc1, desc2);
    
    ///////////////////////////// 
    // Ottengo i vettori di traslazione
    // IMG_1 pose: TIME : 25027785 POS : -->4.467813 3.420069 0.806258<-- 0.074931 -0.160281 0.563678
    // IMG_2 pose: TIME : 25694439 POS : -->5.034858 3.667427 0.833424<-- 0.014587 -0.248119 0.523502
    cv::Vec3d
        translation1(4.467813, 3.420069, 0.806258),
        translation2(5.034858, 3.667427, 0.833424);
        
    ///////////////////////////// 
    // Ottengo i vettori di rotazione (rodrigues)
    // IMG_1 pose: TIME : 25027785 POS : 4.467813 3.420069 0.806258 -->0.074931 -0.160281 0.563678<--
    // IMG_2 pose: TIME : 25694439 POS : 5.034858 3.667427 0.833424 -->0.014587 -0.248119 0.523502<--
    cv::Vec3d
        rodrigues1(0.074931, -0.160281, 0.563678),
        rodrigues2(0.014587, -0.248119, 0.523502);
    
    cv::Matx44d
        g12;
        
    std::vector<cv::Vec3d>
        triagulated /*= cv::Mat::zeros(cv::Size(N,3), CV_64FC1)*/;
        
    std::vector<bool>
        outliersMask; //> Mask to distinguish between inliers (1) and outliers (0) match points. 
                      //> As example those with negative z are outliers.
        
    SingleCameraTriangulator 
        sct(fs);
        
    sct.setKeypoints(kpts1, kpts2, matches);
    sct.setg12(translation1, translation2, rodrigues1, rodrigues2, g12);
    sct.triangulate(triagulated, outliersMask); // 3D points are all inliers! The mask is for the matches
    
    ///////////////////////////// 
    // Visualizzo/salvo i match
    cv::Mat
        window;
    std::vector< cv::Scalar >
        colors;
    drawMatches(img1, img2, window, kpts1, kpts2, matches, colors, outliersMask);
    
    cv::imwrite("matches.pgm", window);
    
    ////////////////////////////
    // Obtain normals by minimization of the intensity residuals of reprojected neighborhood of matches
        
    NormalOptimizer no(fs, &sct);
    
    std::vector<cv::Vec3d>
        normalsVector;
        
    no.setImages(img1, img2);
    
    no.startVisualizerThread();
    
    no.computeOptimizedNormals(triagulated, normalsVector, colors);
    
    std::vector<cv::Matx44d>
        featuresFrames;
        
    no.computeFeaturesFrames(triagulated, normalsVector, featuresFrames);
    
    ////////////////////////////
    // Compute neighborhoods
    NeighborhoodsGenerator
    ng(fs);
    
    std::vector< std::vector<cv::Vec3d> >
        neighborhoodsVector;
    
    ng.computeSquareNeighborhoodsByNormals(featuresFrames, neighborhoodsVector);
    
    std::vector< cv::Mat >
        patchesVector,
        imagePointsVector;
        
    sct.setImages(img1,img2);
    sct.projectPointsToImage(image1, neighborhoodsVector, patchesVector, imagePointsVector);
    
    // Draw the patches and save the image
    cv::Mat
        img1_points;
    drawBackProjectedPoints(img1, img1_points, imagePointsVector, colors);
    
    cv::imwrite("projectedPatches.pgm", img1_points);
    
    ////////////////////////////
    // Converto i punti in point cloud e visualizzo la cloud
    
    pcl::PointCloud<pcl::Normal>::Ptr
        normalsCloud (new pcl::PointCloud<pcl::Normal>());
    
    for (std::vector<cv::Vec3d>::iterator it = normalsVector.begin(); it != normalsVector.end(); it++)
    {
        pcl::Normal
            normal((*it)[0],(*it)[1],(*it)[2]);
            
        normalsCloud->points.push_back(normal);
    }
    no.stopVisualizerThread();
    
//     viewPointCloudNormalsAndFrames(triagulated, normalsCloud, colors, featuresFrames);
    
//     viewPointCloudNormalsFramesAndNeighborhood(neighborhoodsVector, normalsVector, colors, featuresFrames);
    
    cv::Vec3d gravity(no.getGravity());
    
    viewPointCloudNormalsFramesNeighborhoodAndGravity(neighborhoodsVector, normalsVector, colors, featuresFrames, gravity);
    
    return 0;
}
