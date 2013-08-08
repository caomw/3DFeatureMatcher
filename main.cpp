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
#include "Triangulator/neighborhoodsgenerator.h"
#include "tools.h"

#define METODO_A_  // pixel dall'immagine
// #define METODO_B_  // punti 3D dal piano (deprecato, non supporta la versione piramidale ancora)

#define IMG_1 "/home/mpp/WorkspaceTesi/loop_dataset/Images/img_0000000750.pgm"
#define IMG_2 "/home/mpp/WorkspaceTesi/loop_dataset/Images/img_0000000770.pgm"

// IMG_1 pose: TIME : 25027785 POS : 4.467813 3.420069 0.806258 0.074931 -0.160281 0.563678
// IMG_2 pose: TIME : 25694439 POS : 5.034858 3.667427 0.833424 0.014587 -0.248119 0.523502

typedef struct {
    SingleCameraTriangulator *sct;
    NeighborhoodsGenerator *ng;
    cv::Vec3d *point;
    int m_dat;
    double scale;
    std::vector<Pixel> *imagePoints1;
} dataStruct;

void evaluateNormal( const double *par, int m_dat,
                     const void *data, double *fvec,
                     int *info )
{
    dataStruct 
        *D = (dataStruct*) data;
    cv::Vec3d 
        normal(*par,*(par+1),*(par+2));
        
#ifdef METODO_A_
    std::vector<cv::Vec3d>
        pointGroup;
    std::vector<Pixel>
        imagePoints2;
        
//     D->sct->extractPixelsContourAndGet3DPoints(*(D->point), normal, imagePoints1, pointGroup);
//     D->sct->projectPointsToImage2(pointGroup, imagePoints2);
    D->sct->get3dPointsFromImage1Pixels(*(D->point), normal, *(D->imagePoints1), pointGroup);
    D->sct->projectPointsToImage2(pointGroup, D->scale, imagePoints2);
    
    for (std::size_t i = 0; i < m_dat; i++)
    {
        fvec[i] = D->imagePoints1->at(i).i_ - imagePoints2.at(i).i_;
    }
#endif // METODO_A_

#ifdef METODO_B_
    cv::Mat
        neighborhood;
    D->ng->computeNeighborhoodByNormal(*(D->point), normal, neighborhood);
    cv::Mat
        imagePoints1, imagePoints2;
    std::vector<double> 
        residuals;
    D->sct->projectPointsAndComputeResidual(neighborhood, imagePoints1, imagePoints2, residuals);
    for (std::size_t i = 0; i < m_dat; i++)
    {
        fvec[i] = residuals.at(i);
    }
#endif // METODO_B_
}

cv::Vec3d optimize(const cv::Mat &img1, const cv::Mat &img2, double scale, cv::Vec3d &initialNormal, dataStruct &data)
{
    /* parameter vector */
    int n_par = 3;  /* number of parameters in model function f */
    
    double par[3] = { initialNormal[0], initialNormal[1], initialNormal[2] };   
    
    /* auxiliary parameters */
    lm_status_struct status;
    lm_control_struct control = lm_control_double;
    control.epsilon = 1.e-8;
    lm_princon_struct princon = lm_princon_std;
    princon.flags = 3;
    
    data.sct->setImages(img1,img2);
    
    lmmin( n_par, par, data.m_dat, (const void*) &data, evaluateNormal,
           lm_printout_std, &control, &princon, &status );
    
    initialNormal[0] = par[0];
    initialNormal[1] = par[1];
    initialNormal[2] = par[2];
    
    return initialNormal;
}

cv::Vec3d optimizePyramid(const cv::Mat &img1, const cv::Mat &img2, int pyr_levels, cv::Vec3d &initialNormal, dataStruct &data)
{
    cv::Mat 
        img_pyr1[pyr_levels + 1],
        img_pyr2[pyr_levels + 1];
    img_pyr1[0] = img1;
    img_pyr2[0] = img2;
    
    for( int i = 1; i <= pyr_levels; i++)
    {
        cv::pyrDown(img_pyr1[i-1], img_pyr1[i]);
        cv::pyrDown(img_pyr2[i-1], img_pyr2[i]);
    }
    
    float img_scale = float( pow(2.0,double(pyr_levels)) );
    
    for( int i = pyr_levels; i >= 0; i--)
    {
        data.scale = 1.0/img_scale;
        initialNormal = optimize(img_pyr1[i], img_pyr2[i], img_scale, initialNormal, data);
        img_scale /= 2.0f;
    }
    
    return initialNormal;
}

/** Displays the usage message
 */
int help(void)
{
    std::cout << "Usage: 3dfeaturematcher -s <settings.yml>" << std::endl;
    return 0;
}

int main(int argc, char **argv) {
    
    std::cout << "Hello!" << std::endl;
    
    std::cout << std::fixed << std::setprecision(6);
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
        
    cv::Mat
        triagulated /*= cv::Mat::zeros(cv::Size(N,3), CV_64FC1)*/;
        
    std::vector<bool>
        outliersMask; //> Mask to distinguish between inliers (1) and outliers (0) points. As example those with negative z are outliers.
        
    SingleCameraTriangulator 
        sct(fs);
        
    sct.setKeypoints(kpts1, kpts2, matches);
    sct.setg12(translation1, translation2, rodrigues1, rodrigues2, g12);
    sct.triangulate(triagulated, outliersMask);
    
//     std::cout << triagulated << std::endl;
    
    ///////////////////////////// 
    // Visualizzo/salvo le immagini e i match
    cv::Mat
        window;
    std::vector< cv::Scalar >
        colors;
    drawMatches(img1, img2, window, kpts1, kpts2, matches, colors, outliersMask);
    
    cv::imwrite("matches.pgm", window);
    
    ////////////////////////////
    // Ottengo le guess delle normali dei piani e un set di punti che campionano gli intorni dei punti triangolati
    NeighborhoodsGenerator
        ng(fs);
        
//     std::vector<cv::Vec3d>
//         normalsVector;
    pcl::PointCloud<pcl::Normal>::Ptr
        normalsCloud (new pcl::PointCloud<pcl::Normal>());
        
    std::vector<cv::Mat>
        neighborhoodsVector;
        
    std::vector<cv::Mat>
        imagePointsVector1,
        imagePointsVector2;
    
    std::vector< std::vector<double> >
        residualsVectors;
        
    sct.setImages(img1, img2);
    
#ifdef METODO_A_
    int m_dat = sct.getMdat();
    int pyr_levels;
    fs["Neighborhoods"]["pyramids"] >> pyr_levels;
#endif // METODO_A_

#ifdef METODO_B_
    int
        numberOfRays, numberOfThetas;
    
    fs["Neighborhoods"]["rays"] >> numberOfRays;
    fs["Neighborhoods"]["thetas"] >> numberOfThetas;
    
    int
        m_dat = numberOfRays * numberOfThetas;
#endif // METODO_B_
        
    for (std::size_t actualPointIndex = 0; actualPointIndex < triagulated.cols; actualPointIndex++)
    {
        
        /* data points */
        cv::Vec3d 
            point = triagulated.col(actualPointIndex),
            normal = (-1) * point / cv::norm(point); /* the initial guess */
        
        std::vector<Pixel>
            imagePoints1;
            
        sct.extractPixelsContour(point, 1.0, imagePoints1);
            
        dataStruct data = { &sct, &ng, &point, m_dat, 1.0, &imagePoints1};
        
        cv::Vec3d newNormal = optimizePyramid(img1, img2, pyr_levels, normal, data);
        
        std::cout << normal-newNormal << std::endl;
        
        pcl::Normal
            n(newNormal[0], newNormal[1], newNormal[2]);
            
        normalsCloud->push_back(n);
        
//         cv::Mat
//             neighborhood;
//         cv::Vec3d
//             point = triagulated.col(actualPointIndex),
//             normal;
//         ng.computeNeighborhoodByNormal(point, normal, neighborhood);
//         normalsVector.push_back(normal);
//         neighborhoodsVector.push_back(neighborhood);
//         
//         cv::Mat
//             imagePoints1, imagePoints2;
//         std::vector<double>
//             residuals;
//         sct.projectPointsAndComputeResidual(neighborhood, imagePoints1, imagePoints2, residuals);
//         residualsVectors.push_back(residuals);
//         imagePointsVector1.push_back(imagePoints1);
//         imagePointsVector2.push_back(imagePoints2);
//         
//         for (std::vector<double>::iterator it = residuals.begin(); it != residuals.end(); it++)
//         {
//             std::cout << (*it) << " - ";
//         }
//         std::cout << std::endl;
    }
    
//     cv::Mat
//         test1, test2, img1_BGR, img2_BGR;
//         
//     cv::cvtColor(img1, img1_BGR, CV_GRAY2BGR);
//     cv::cvtColor(img2, img2_BGR, CV_GRAY2BGR);
//     
//     drawBackProjectedPoints(img1_BGR, test1, imagePointsVector1, colors);
//     drawBackProjectedPoints(img2_BGR, test2, imagePointsVector2, colors);
//     
//     cv::imwrite("test1.pgm", test1);
//     cv::imwrite("test2.pgm", test2);
    
//     cv::namedWindow("test1");
//     cv::imshow("test1", test1);
//     cv::namedWindow("test2");
//     cv::imshow("test2", test2);
//     cv::waitKey();
    
    ///////////////////////////// 
    // Converto i punti in point cloud e visualizzo la cloud
//     viewPointCloud(triagulated, colors);
    
//     viewPointCloudNeighborhood(triagulated, neighborhoodsVector, colors);
//     viewPointCloudAndNormals(triagulated, normalsCloud, colors);
    
    return 0;
}
