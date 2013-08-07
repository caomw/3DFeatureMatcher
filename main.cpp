#include <iostream>
#include <locale>
#include <fstream>
#include <iomanip>
#include <vector>
#include <math.h>

#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "DescriptorsMatcher/descriptorsmatcher.h"
#include "Triangulator/singlecameratriangulator.h"
#include "Triangulator/neighborhoodsgenerator.h"
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
        outliersMask; //> Mask to distinguis between inliers (1) and outliers (0) points. As example those with negative z are outliers.
        
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
    
//     // Lavoro su un solo punto per fare i test:
//     cv::Mat
//         triagulated_subvector = cv::Mat::zeros(cv::Size(1,3), CV_64FC1);
//     triagulated_subvector.at<cv::Vec3d>(0) = triagulated.at<cv::Vec3d>(0);
    
    NeighborhoodsGenerator
        ng(fs);
        
    cv::Mat
        normals;
        
    std::vector<cv::Mat>
        neighborhoodsVector;
        
    ng.computeNeighborhoodsByNormals(triagulated, normals, neighborhoodsVector);

    ///////////////////////////// 
    // Proietto i punti sulle 2 immagini e calcolo il residuo
    
    double
        Fx, Fy, Cx, Cy;
    
    fs["CameraSettings"]["Fx"] >> Fx;
    fs["CameraSettings"]["Fy"] >> Fy;
    fs["CameraSettings"]["Cx"] >> Cx;
    fs["CameraSettings"]["Cy"] >> Cy;
    
    cv::Matx33d
        cameraMatrix;

    cameraMatrix(0,0) = Fx;
    cameraMatrix(1,1) = Fy;
    cameraMatrix(0,2) = Cx;
    cameraMatrix(1,2) = Cy;
    cameraMatrix(2,2) = 1;
    
    double
        p1, p2, k0, k1, k2 ;
    
    fs["CameraSettings"]["p1"] >> p1;
    fs["CameraSettings"]["p2"] >> p2;
    fs["CameraSettings"]["k0"] >> k0;
    fs["CameraSettings"]["k1"] >> k1;
    fs["CameraSettings"]["k2"] >> k2;
    
    cv::Mat distortionCoefficients = cv::Mat(cv::Size(5,1), CV_64FC1, cv::Scalar(0));
    
    distortionCoefficients.at<double>(0) = k0;
    distortionCoefficients.at<double>(1) = k1;
    distortionCoefficients.at<double>(2) = p1;
    distortionCoefficients.at<double>(3) = p2;
    distortionCoefficients.at<double>(4) = k2;
    
    cv::Mat
        t1 = cv::Mat::zeros(cv::Size(3,1),cv::DataType<float>::type), 
        r1 = cv::Mat::zeros(cv::Size(3,1),cv::DataType<float>::type);
        
    cv::Vec3d 
        t2, r2;
        
    decomposeTransformation(g12.inv(), r2, t2);
    
    std::cout << g12 << std::endl << g12.inv() << std::endl;
    
    cv::Ptr<cv::Mat>
        imagePoints1,
        imagePoints2;
        
    std::vector<cv::Mat>
        imagePointsVector1,
        imagePointsVector2;
    
    for (std::size_t actualNeighborIndex = 0; actualNeighborIndex < neighborhoodsVector.size(); actualNeighborIndex++)
    {    
        imagePoints1 = new cv::Mat(cv::Size(1,1), CV_64FC2, cv::Scalar(0,0));
        imagePoints2 = new cv::Mat(cv::Size(1,1), CV_64FC2, cv::Scalar(0,0));
        
        cv::projectPoints(neighborhoodsVector.at(actualNeighborIndex), t1, r1, cameraMatrix, distortionCoefficients, *imagePoints1);
        cv::projectPoints(neighborhoodsVector.at(actualNeighborIndex), t2, r2, cameraMatrix, distortionCoefficients, *imagePoints2);
        
//         std::cout << neighborhoodsVector.at(actualNeighborIndex) << std::endl;
//         std::cout << imagePoints << std::endl;
        
        imagePointsVector1.push_back(*imagePoints1);
        imagePointsVector2.push_back(*imagePoints2);
    }
    
    cv::Mat
        test1, test2, img1_BGR, img2_BGR;
        
    cv::cvtColor(img1, img1_BGR, CV_GRAY2BGR);
    cv::cvtColor(img2, img2_BGR, CV_GRAY2BGR);
    
    drawBackProjectedPoints(img1_BGR, test1, imagePointsVector1, colors);
    drawBackProjectedPoints(img2_BGR, test2, imagePointsVector2, colors);
    
    cv::imwrite("test1.pgm", test1);
    cv::imwrite("test2.pgm", test2);
    
    cv::namedWindow("test1");
    cv::imshow("test1", test1);
    cv::namedWindow("test2");
    cv::imshow("test2", test2);
    cv::waitKey();
    
    ///////////////////////////// 
    // Converto i punti in point cloud e visualizzo la cloud
//     viewPointCloud(triagulated, colors);
    
    viewPointCloudNeighborhood(triagulated, neighborhoodsVector, colors);
    
    return 0;
}
