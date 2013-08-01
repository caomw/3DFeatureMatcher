#include <iostream>
#include <locale>
#include <fstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/lexical_cast.hpp>
#include <boost/thread/thread.hpp>

#define IMG_1 "/home/mpp/WorkspaceTesi/loop_dataset/Images/img_0000000750.pgm"
#define IMG_2 "/home/mpp/WorkspaceTesi/loop_dataset/Images/img_0000000770.pgm"

// IMG_1 pose: TIME : 25027785 POS : 4.467813 3.420069 0.806258 0.074931 -0.160281 0.563678
// IMG_2 pose: TIME : 25694439 POS : 5.034858 3.667427 0.833424 0.014587 -0.248119 0.523502

void composeTransformation(const cv::Matx33d &R, const cv::Vec3d &T, cv::Matx44d &G)
{
    // Put R on top left
    G(0,0) = R(0,0); G(0,1) = R(0,1); G(0,2) = R(0,2);
    G(1,0) = R(1,0); G(1,1) = R(1,1); G(1,2) = R(1,2);
    G(2,0) = R(2,0); G(2,1) = R(2,1); G(2,2) = R(2,2);
    
    // Put T' on top right
    G(0,3) = T(0); G(1,3) = T(1); G(2,3) = T(2);
    
    // Put homogeneous variables
    G(3,0) = 0; G(3,1) = 0; G(3,2) = 0; G(3,3) = 1;
}

int main(int argc, char **argv) {
    
    std::cout << "Hello!" << std::endl;
    
    std::cout << std::fixed << std::setprecision(6);
    setlocale(LC_NUMERIC, "C");
    
    // Get images
    cv::Mat
        img1 = cv::imread(IMG_1, CV_LOAD_IMAGE_GRAYSCALE),
        img2 = cv::imread(IMG_2, CV_LOAD_IMAGE_GRAYSCALE);
    
    ////
    // Ottengo i vettori di traslazione
    // IMG_1 pose: TIME : 25027785 POS : -->4.467813 3.420069 0.806258<-- 0.074931 -0.160281 0.563678
    // IMG_2 pose: TIME : 25694439 POS : -->5.034858 3.667427 0.833424<-- 0.014587 -0.248119 0.523502
    // Tc_x : 0.0
    // Tc_y : 0.015
    // Tc_z : -0.051
    cv::Vec3d
        translation1(4.467813, 3.420069, 0.806258),
        translation2(5.034858, 3.667427, 0.833424),
        translationIC(0.0, 0.015, -0.051);
        
    ////
    // Ottengo le matrici 3x3 di rotazione
    // IMG_1 pose: TIME : 25027785 POS : 4.467813 3.420069 0.806258 -->0.074931 -0.160281 0.563678<--
    // IMG_2 pose: TIME : 25694439 POS : 5.034858 3.667427 0.833424 -->0.014587 -0.248119 0.523502<--
    // Wc_x : -1.2005
    // Wc_y : 1.1981
    // Wc_z : 1.2041
    cv::Vec3d
        rodrigues1(0.074931, -0.160281, 0.563678),
        rodrigues2(0.014587, -0.248119, 0.523502),
        rodriguesIC(-1.2005, 1.1981, 1.2041);
      
    cv::Matx33d
        rotation1, rotation2, rotationIC;
        
    cv::Rodrigues(rodrigues1, rotation1);
    cv::Rodrigues(rodrigues2, rotation2);
    cv::Rodrigues(rodriguesIC, rotationIC);
    
//     std::cout << "-Rotazioni-" << std::endl;
//     std::cout << rotationIC << std::endl;
//     std::cout << "------------" << std::endl;
    
    ////
    // Preparo le matrici di trasformazione
    cv::Matx44d
        g1, g2, g12, gIC;  //> gIC = Matrice di trasformazione da CAMERA a IMU
        
    composeTransformation(rotation1, translation1, g1);
    composeTransformation(rotation2, translation2, g2);
    composeTransformation(rotationIC, translationIC, gIC);
        
    std::cout << "-Trasformazioni-" << std::endl;
    std::cout << g1 << std::endl << g2 << std::endl << gIC << std::endl;
    
    // Calcolo la trasformazione dal CAMERA2 a CAMERA1
    cv::Matx44d
        g1C, g2C;
    
    g12 = gIC * g1.inv() * g2 * gIC.inv();
    
    std::cout << "-g12-" << std::endl;
    std::cout << g12 << std::endl;
    std::cout << "------------" << std::endl;
    
    // Setup camera matrix and the distortion coefficients
    /* Camera matrix:
        Fx 0  Cx
        0  Fx Cy
        0  0  1 */
    double
        Fx = 572.4765,
        Fy = 572.69354,
        Cx = 549.75189,
        Cy = 411.68039;
        
    cv::Matx33d
        cameraMatrix;
    
    cameraMatrix(0,0) = Fx;
    cameraMatrix(1,1) = Fy;
    cameraMatrix(0,2) = Cx;
    cameraMatrix(1,2) = Cy;
    
    std::cout << "-Camera matrix-" << std::endl;
    std::cout << cameraMatrix << std::endl;
    std::cout << "------------" << std::endl;
    
    /*
    distCoeffs:
    */
    double
        p1 = -6.6e-05,
        p2 = 0.000567,
        k0 = -0.299957,
        k1 = 0.124129,
        k2 = -0.028357;
        
    cv::Mat
        distCoeffs = cv::Mat::zeros(cv::Size(5,1), CV_64FC1);
        
    distCoeffs.at<double>(0) = k0;
    distCoeffs.at<double>(1) = k1;
    distCoeffs.at<double>(2) = p1;
    distCoeffs.at<double>(3) = p2;
    distCoeffs.at<double>(4) = k2;
    
    ////
    // Prendo i punti, li rettifico e li triangolo
    cv::Matx34d
        projection1, projection2;
    projection1 = projection1.eye(); // eye() da solo non modifica la matrice
    projection2 = projection2.eye() * g12;
    
    std::cout << "-Proiezioni-" << std::endl;
    std::cout << projection1 << std::endl << projection2 << std::endl;
    std::cout << "------------" << std::endl;

    int
        N = 2;  //> Numero di punti da triangolare
        
    cv::Mat
        a1 = cv::Mat::zeros(cv::Size(1,N), CV_64FC2),   //> Punti sulla prima immagine
        a2 = cv::Mat::zeros(cv::Size(1,N), CV_64FC2),   //> Punti sulla seconda immagine
        ua1, ua2;   //> Punti rettificati
        
    ////
    // Punto sul cartellone
    // (347,403)
    // (204,372)
    a1.at<cv::Vec2d>(0) = cv::Vec2d(347,403);
    a2.at<cv::Vec2d>(0) = cv::Vec2d(204,372);
        
    ////
    // Punto sul libro bianco sulla libreria
    // (781,588)
    // (804,695)
    a1.at<cv::Vec2d>(1) = cv::Vec2d(781,588);
    a2.at<cv::Vec2d>(1) = cv::Vec2d(804,695);
    
    cv::undistortPoints(a1, ua1, cameraMatrix, distCoeffs);
    cv::undistortPoints(a2, ua2, cameraMatrix, distCoeffs);
    
    std::cout << "-Punti originali-" << std::endl;
    std::cout << a1 << std::endl << a2 << std::endl;
    std::cout << "------------" << std::endl;
    std::cout << "-Punti rettificati-" << std::endl;
    std::cout << ua1 << std::endl << ua2 << std::endl;
    std::cout << "------------" << std::endl;
    
    cv::Mat
        triagulatedHomogeneous = cv::Mat::zeros(cv::Size(N,4), CV_64FC1),
        triagulated = cv::Mat::zeros(cv::Size(N,3), CV_64FC1);
        
    cv::triangulatePoints(projection1, projection2, ua1, ua2, triagulatedHomogeneous);
    
    // Get cartesian coordinate from homogeneous
    for (int actualPoint = 0; actualPoint < N; actualPoint++)
    {
        double 
            X = triagulatedHomogeneous.at<double>(0, actualPoint),
            Y = triagulatedHomogeneous.at<double>(1, actualPoint),
            Z = triagulatedHomogeneous.at<double>(2, actualPoint),
            W = triagulatedHomogeneous.at<double>(3, actualPoint);
            
//         std::cout << "[" << X << ", " << Y << ", " << Z << ", " << W << "]" << std::endl;
//         std::cout << "[" << X/W << ", " << Y/W << ", " << Z/W << ", " << W/W << "]" << std::endl;
        
        triagulated.at<double>(0, actualPoint) = X / W; // get x = X/W
        triagulated.at<double>(1, actualPoint) = Y / W; // get y = Y/W
        triagulated.at<double>(2, actualPoint) = Z / W; // get z = Z/W
    }
    
    std::cout << "-Punti triangolati in coordinate omogenee (matrice 4xN)-" << std::endl;
    std::cout << triagulatedHomogeneous << std::endl;
    std::cout << "------------" << std::endl;
    std::cout << "-Punti triangolati (matrice 3xN)-" << std::endl;
    std::cout << triagulated << std::endl;
    std::cout << "------------" << std::endl;
    
    ////
    // Converto i punti in point cloud
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        
    for (int i = 0; i < N; i++)
    {
        pcl::PointXYZRGB actual;
        actual.x = triagulated.at<double>(0, i);
        actual.y = triagulated.at<double>(1, i);
        actual.z = triagulated.at<double>(2, i);
        actual.r = (i%2)?255:100;
        actual.g = (i%2)?100:100;
        actual.b = (i%2)?100:255;
        
        cloud->points.push_back(actual);
    }
    cloud->width = (int) cloud->points.size ();
    cloud->height = 1;
    
    ////
    // Visualizzo la point cloud
    boost::shared_ptr< pcl::visualization::PCLVisualizer >
        viewer( new pcl::visualization::PCLVisualizer("Triangulated points viewer") );
        
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Triangulated points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "Triangulated points");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
        
    cv::namedWindow("test1");
    cv::namedWindow("test2");
    cv::imshow("test1", img1);
    cv::imshow("test2", img2);
    cv::waitKey();
    
    while (!viewer->wasStopped())
    {
        viewer->spin();
    }
    
    return 0;
}
