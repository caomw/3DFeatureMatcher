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
#include "pclvisualizerthread.h"
#include "tools.h"

#define IMG_1 "/home/mpp/WorkspaceTesi/loop_dataset/Images/img_0000000750.pgm"
#define IMG_2 "/home/mpp/WorkspaceTesi/loop_dataset/Images/img_0000000770.pgm"

// IMG_1 pose: TIME : 25027785 POS : 4.467813 3.420069 0.806258 0.074931 -0.160281 0.563678
// IMG_2 pose: TIME : 25694439 POS : 5.034858 3.667427 0.833424 0.014587 -0.248119 0.523502

// // Support struct to minimization process
// typedef struct {
//     
//     SingleCameraTriangulator 
//         *sct;
//     cv::Vec3d 
//         *point;
//     int 
//         m_dat;
//     double 
//         scale;
//     std::vector<Pixel> 
//         *imagePoints1;
//         
// } dataStruct;


// // Function that evaluate fvec of lmmin
// void evaluateNormal( const double *par, int m_dat,
//                      const void *data, double *fvec,
//                      int *info )
// {
//     dataStruct 
//         *D = (dataStruct*) data;
//     cv::Vec3d 
//         normal;
//         
//     double
//         theta = par[0],
//         phi = par[1];
// 
//     // rho = 1
//     // x = [rho * sin(theta)] * cos(phi)
//     normal[0] = cos(abs(theta)) * sin(phi);
//     // y = [rho * sin(theta)] * sin(phi)
//     normal[1] = sin(theta);
//     // z = rho * cos(theta)
//     normal[2] = cos(abs(theta)) * cos(phi);
//     
//     // lmmin can go over 1 for normal coordinate, which is wrong for a normal versor.
//     if (isnan(normal[2]) || isnan(normal[1]) || isnan(normal[0]))
//     {
//         std::cout << "male" << std::endl;
//     }
//         
//     std::vector<cv::Vec3d>
//         pointGroup;
//     std::vector<Pixel>
//         imagePoints2;
//         
//     // obtain 3D points
//     D->sct->get3dPointsFromImage1Pixels(*(D->point), normal, *(D->imagePoints1), pointGroup);
//     
//     cv::Vec3d a = pointGroup[0];
//     // update imagePoints1 to actual scale pixels intensity
//     D->sct->updateImage1PixelsIntensity(D->scale, *(D->imagePoints1));
//     // get imagePoints2 at actual scale
//     D->sct->projectPointsToImage2(pointGroup, D->scale, imagePoints2);
//    
// 
// //     if (D->scale == 1.0)
// //     {
// //         viewPointCloud(pointGroup, normal);
// //     }
//     
//     for (std::size_t i = 0; i < m_dat; i++)
//     {
//         fvec[i] = D->imagePoints1->at(i).i_ - imagePoints2.at(i).i_;
//     }
//     
// //     if (D->scale == 1.0)
// //     {
// //         for (std::size_t i = 0; i < m_dat; i++)
// //         {
// //             std::cout <<  D->imagePoints1->at(i).i_ - imagePoints2.at(i).i_ << " - ";
// //         }
// //         std::cout << std::endl;
// //     }
// }
// 
// void computePyramids(const cv::Mat &img1, const cv::Mat &img2, const int pyr_levels, std::vector<cv::Mat> &img_pyr1, std::vector<cv::Mat> &img_pyr2)
// {
//     cv::Mat 
//         pyr1 = img1, pyr2 = img2;
//         
//     img_pyr1.push_back(pyr1);
//     img_pyr2.push_back(pyr2);
//     
//     for( int i = 1; i <= pyr_levels; i++)
//     {
//         cv::pyrDown(img_pyr1[i-1], pyr1);
//         cv::pyrDown(img_pyr2[i-1], pyr2);
//         img_pyr1.push_back(pyr1);
//         img_pyr2.push_back(pyr2);
//     }
// }
// 
// cv::Vec3d optimize(const cv::Mat &img1, const cv::Mat &img2, cv::Vec3d &initialNormal, dataStruct &data, double epsilon)
// {
//     // convert the normal to spherical coordinates
//     double 
//         theta, phi;
// 
//     theta = asin(initialNormal[1]); // elevation
//     phi = atan(initialNormal[0]/initialNormal[2]); // azimuth
//     
//     /* parameter vector */
//     int 
//         n_par = 2;  // number of parameters in evaluateNormal
//     double 
//         par[2] = { phi, theta };   
//     
//     /* auxiliary parameters */
//     lm_status_struct 
//         status;
//         
//     lm_control_struct 
//         control = lm_control_double;
//     control.epsilon = epsilon;
//     
//     lm_princon_struct 
//         princon = lm_princon_std;
//     princon.flags = 0;
//     
//     data.sct->setImages(img1,img2);
//     
//     lmmin( n_par, par, data.m_dat, (const void*) &data, evaluateNormal,
//            lm_printout_std, &control, 0/*&princon*/, &status );
//     
//     cv::Vec3d
//         estimatedNormal;
//         
//     theta = par[0],
//     phi = par[1];
//     
//     // rho = 1
//     // x = [rho * sin(theta)] * cos(phi)
//     estimatedNormal[0] = cos(abs(theta)) * sin(phi);
//     // y = [rho * sin(theta)] * sin(phi)
//     estimatedNormal[1] = sin(theta);
//     // z = rho * cos(theta)
//     estimatedNormal[2] = cos(abs(theta)) * cos(phi);
//     
//     return estimatedNormal;
// }
// 
// cv::Vec3d optimizePyramid(const std::vector<cv::Mat> &img_pyr1, const std::vector<cv::Mat> &img_pyr2, 
//                           int pyr_levels, cv::Vec3d &initialNormal, dataStruct &data, double epsilon)
// {
//     float img_scale = float( pow(2.0,double(pyr_levels)) );
//     
//     cv::Vec3d
//         estimatedNormal = initialNormal; 
//     
//     for( int i = pyr_levels; i >= 0; i--)
//     {
//         data.scale = 1.0/img_scale;
//         estimatedNormal = optimize(img_pyr1[i], img_pyr2[i], estimatedNormal, data, epsilon);
//         img_scale /= 2.0f;
//     }
//     
//     return estimatedNormal;
// }

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
    
    no.computeOptimizedNormals(triagulated, normalsVector);
    
    ///////////////////////////// 
    // Converto i punti in point cloud e visualizzo la cloud
    
    pcl::PointCloud<pcl::Normal>::Ptr
        normalsCloud (new pcl::PointCloud<pcl::Normal>());
    
    for (std::vector<cv::Vec3d>::iterator it = normalsVector.begin(); it != normalsVector.end(); it++)
    {
        pcl::Normal
            normal((*it)[0],(*it)[1],(*it)[2]);
            
        normalsCloud->points.push_back(normal);
    }
        
    viewPointCloudAndNormals(triagulated, normalsCloud, colors);
    
    return 0;
}
