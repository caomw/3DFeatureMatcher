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
        
    cv::Mat
        triagulated /*= cv::Mat::zeros(cv::Size(N,3), CV_64FC1)*/;
        
    std::vector<bool>
        outliersMask; //> Mask to distinguis between inliers (1) and outliers (0) points. As example those with negative z are outliers.
        
    SingleCameraTriangulator 
        sct(fs);
        
    sct.setKeypoints(kpts1, kpts2, matches);
    sct.setg12(translation1, translation2, rodrigues1, rodrigues2);
    sct.triangulate(triagulated, outliersMask);
    
    std::cout << triagulated << std::endl;
    
    ///////////////////////////// 
    // Visualizzo/salvo le immagini e i match
    cv::Mat
        window;
    std::vector< cv::Scalar >
        colors;
    drawMatches(img1, img2, window, kpts1, kpts2, matches, colors, outliersMask);
    
    cv::imwrite("matches.pgm", window);
    
    ////////////////////////////
    // Ottengo le guess delle normali dei piani 
    
    // Lavoro su un sotto vettore:
//     cv::Mat
//         triagulated_subvector = cv::Mat::zeros(cv::Size(1,3), CV_64FC1);
//     triagulated_subvector.at<cv::Vec3d>(0) = triagulated.at<cv::Vec3d>(0);
    
    std::vector<cv::Vec3d>
        normals;
        
    std::vector< std::vector<cv::Vec3d> >
        neighborhoodsVector;
    
    const double
        epsilon = 0.15; //> Take a neighborhood of 0.3m around each point
        
    const double
        thetaIncrement = 2*M_PI / 10;
    
    for (std::size_t actualPoint = 0; actualPoint < triagulated.cols; actualPoint++)
    {
        cv::Vec3d
            point = triagulated.col(actualPoint),//.at<cv::Vec3d>(actualPoint),
            normal;
            
        normal = point / cv::norm(point);        
        normals.push_back(normal);
        
        // Compute a perpendicular vector
        cv::Vec3d
            spanner(0,1,-normal[1]/normal[2]); // in this way <spanner, normal> = 0
            
        std::cout << "<" << normal << ", " << spanner << "> = " << normal.dot(spanner) << std::endl;
        
        spanner = spanner / cv::norm(spanner) * epsilon;
        
        // Compute the neighborhood of each point
        std::vector<cv::Vec3d>
            neighborhood;
        
        for (double r = 0.1; r <= 1; r = r + 0.1)
        {
            for (double theta = 0; theta < 2*M_PI; theta = theta + thetaIncrement)
            {
                // Using the rodrigues formula construct the rotation matrix
                cv::Matx33d
                    W, I, R;
                W(0,0) = 0;          W(0,1) = -normal[2]; W(0,2) =  normal(1); 
                W(1,0) =  normal[2]; W(1,1) = 0;          W(1,2) = -normal[0]; 
                W(2,0) = -normal[1]; W(2,1) =  normal[0]; W(2,2) = 0; 
                
//                 R = I.eye() + sin(theta)*W + (2 * (sin(theta/2)) * (sin(theta/2))) * W.;
                
                cv::Vec3d
                    spannedPoint;
                    
                spannedPoint = point + r *(spanner + W*spanner*sin(theta) + (2 * (sin(theta/2)) * (sin(theta/2))) * W * W * spanner);
                
                std::cout << point << " - " << spannedPoint << std::endl;
                
                neighborhood.push_back(spannedPoint);
            }
        }
        
        std::cout << neighborhood.size() << std::endl;
        
        neighborhoodsVector.push_back(neighborhood);
    }
    
//     std::cout << neighborhoodsVector.size() << " - " << triagulated.cols << std::endl;
    
    ///////////////////////////// 
    // Converto i punti in point cloud e visualizzo la cloud
//     viewPointCloud(triagulated, colors);
    
    viewPointCloudNeighborhood(triagulated, neighborhoodsVector, colors);
    
    return 0;
}
