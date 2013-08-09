/**
 * \file singlecameratriangulator.cpp
 * \Author: Michele Marostica
 *  
 * Copyright (c) 2012, Michele Marostica (michelemaro@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 * 1 - Redistributions of source code must retain the above copyright notice, 
 * 	   this list of conditions and the following disclaimer.
 * 2 - Redistributions in binary form must reproduce the above copyright 
 *	   notice, this list of conditions and the following disclaimer in the
 *	   documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <math.h>

#include "singlecameratriangulator.h"

SingleCameraTriangulator::SingleCameraTriangulator(cv::FileStorage &settings)
{
    /// Setup the parameters
    
    // 1 Camera-IMU translation
    std::vector<double>
        translationIC_vector;
    
    settings["CameraSettings"]["translationIC"] >> translationIC_vector;
    
    translation_IC_ = new cv::Vec3d(translationIC_vector[0],translationIC_vector[1],translationIC_vector[2]);
    
    // 2 Camera-IMU rotation
    std::vector<double>
        rodriguesIC_vector;
    
        settings["CameraSettings"]["rodriguesIC"] >> rodriguesIC_vector;
        
    cv::Vec3d
        rodriguesIC(rodriguesIC_vector[0],rodriguesIC_vector[1],rodriguesIC_vector[2]);
        
    rotation_IC_ = new cv::Matx33d();
    
    cv::Rodrigues(rodriguesIC, *rotation_IC_);
    
    // 3 Camera-IMU transformation
    g_IC_ = new cv::Matx44d();
    
    composeTransformation(*rotation_IC_, *translation_IC_, *g_IC_);
        
    // 4 Compose the camera matrix
    /*        
     * Fx 0  Cx
     * 0  Fx Cy
     * 0  0  1 
     */
    double
        Fx, Fy, Cx, Cy;
        
    settings["CameraSettings"]["Fx"] >> Fx;
    settings["CameraSettings"]["Fy"] >> Fy;
    settings["CameraSettings"]["Cx"] >> Cx;
    settings["CameraSettings"]["Cy"] >> Cy;
    
    camera_matrix_ = new cv::Matx33d();
    
    (*camera_matrix_)(0,0) = Fx;
    (*camera_matrix_)(1,1) = Fy;
    (*camera_matrix_)(0,2) = Cx;
    (*camera_matrix_)(1,2) = Cy;
    (*camera_matrix_)(2,2) = 1;
    
    // 5 Compose the distortion coefficients array
    double
        p1, p2, k0, k1, k2 ;
    
    settings["CameraSettings"]["p1"] >> p1;
    settings["CameraSettings"]["p2"] >> p2;
    settings["CameraSettings"]["k0"] >> k0;
    settings["CameraSettings"]["k1"] >> k1;
    settings["CameraSettings"]["k2"] >> k2;
    
    distortion_coefficients_ = new cv::Mat(cv::Size(5,1), CV_64FC1, cv::Scalar(0));
    
    distortion_coefficients_->at<double>(0) = k0;
    distortion_coefficients_->at<double>(1) = k1;
    distortion_coefficients_->at<double>(2) = p1;
    distortion_coefficients_->at<double>(3) = p2;
    distortion_coefficients_->at<double>(4) = k2;
        
    // 6 Get the outliers threshold
    settings["CameraSettings"]["zThreshold"] >> z_threshold_;
    
    
    settings["Neighborhoods"]["pixelsRay"] >> pixels_ray_;
    
}

void SingleCameraTriangulator::setImages(const cv::Mat& img1, const cv::Mat& img2)
{
    img_1_ = new cv::Mat(img1);
    img_2_ = new cv::Mat(img2);
}


void SingleCameraTriangulator::setg12(const cv::Vec3d& T1, const cv::Vec3d& T2, const cv::Vec3d& rodrigues1, const cv::Vec3d& rodrigues2, cv::Matx44d &g12)
{
    // Compute CAMERA2 to CAMERA1 transformation
    cv::Matx33d
        rotation1, rotation2;
        
    cv::Rodrigues(rodrigues1, rotation1);
    cv::Rodrigues(rodrigues2, rotation2);
    
    cv::Matx44d
        g1, g2;
    
    composeTransformation(rotation1, T1, g1);
    composeTransformation(rotation2, T2, g2);
    
    g_12_ = new cv::Matx44d();
    
    (*g_12_) = (*g_IC_).inv() * g2.inv() * g1 * (*g_IC_);
    
    g12 = (*g_12_);
}

void SingleCameraTriangulator::setKeypoints(const std::vector< cv::KeyPoint >& kpts1, const std::vector< cv::KeyPoint >& kpts2, const std::vector< cv::DMatch >& matches)
{
    N_ = matches.size();
    
    a_1_ = new cv::Mat(cv::Size(1,N_), CV_64FC2, cv::Scalar(0,0));   //> Punti sulla prima immagine
    a_2_ = new cv::Mat(cv::Size(1,N_), CV_64FC2, cv::Scalar(0,0));   //> Punti sulla seconda immagine
    
    for (int actualMatch = 0; actualMatch < N_; actualMatch++)
    {
        int 
            image1Idx = matches.at(actualMatch).queryIdx,
            image2Idx = matches.at(actualMatch).trainIdx;
        
        cv::Vec2d
            pt1(kpts1.at(image1Idx).pt.x, kpts1.at(image1Idx).pt.y),
            pt2(kpts2.at(image2Idx).pt.x, kpts2.at(image2Idx).pt.y);
        
        a_1_->at<cv::Vec2d>(actualMatch) = pt1;
        a_2_->at<cv::Vec2d>(actualMatch) = pt2;
    }
    
    u_1_ = new cv::Mat(cv::Size(1,N_), CV_64FC2, cv::Scalar(0,0));
    u_2_ = new cv::Mat(cv::Size(1,N_), CV_64FC2, cv::Scalar(0,0));
    
    cv::undistortPoints(*a_1_, *u_1_, *camera_matrix_, *distortion_coefficients_);
    cv::undistortPoints(*a_2_, *u_2_, *camera_matrix_, *distortion_coefficients_);
}

void SingleCameraTriangulator::triangulate(cv::Mat& triangulatedPoints, std::vector<bool> &outliersMask)
{
    // Prepare projection matricies
    projection_1_ = new cv::Matx34d();
    projection_2_ = new cv::Matx34d();
    
    (*projection_1_) = projection_1_->eye(); // eye() da solo non modifica la matrice
    (*projection_2_) = projection_2_->eye() * (*g_12_);
    
    // Triangulate points
    cv::Mat
        triagulatedHomogeneous = cv::Mat::zeros(cv::Size(N_,4), CV_64FC1);
    
    cv::triangulatePoints(*projection_1_, *projection_2_, *u_1_, *u_2_, triagulatedHomogeneous);
    
    for (int actualPoint = 0; actualPoint < N_; actualPoint++)
    {
        double 
            X = triagulatedHomogeneous.at<double>(0, actualPoint),
            Y = triagulatedHomogeneous.at<double>(1, actualPoint),
            Z = triagulatedHomogeneous.at<double>(2, actualPoint),
            W = triagulatedHomogeneous.at<double>(3, actualPoint);
        
        // Drop negative z or too far points
        if ( Z / W < 0 || Z / W >= z_threshold_ )
        {
            outliersMask.push_back(false);
            outliers_mask_.push_back(false);
            // Do not save the point
        }
        else
        {
            outliersMask.push_back(true);
            outliers_mask_.push_back(true);
            
            cv::Vec3d
                point(X / W, Y / W, Z / W);
            
            triagulated_.push_back(point);
        }
    }
    
    int
        size = triagulated_.size();
        
    triangulatedPoints = cv::Mat::zeros(cv::Size(size, 3), CV_64FC1);
    
    for (std::size_t k = 0; k < size; k++)
    {
        triangulatedPoints.at<double>(0, k) = triagulated_[k][0];
        triangulatedPoints.at<double>(1, k) = triagulated_[k][1];
        triangulatedPoints.at<double>(2, k) = triagulated_[k][2];
    }
}

void SingleCameraTriangulator::projectPointsToImages( const std::vector< cv::Mat >& pointsGroupVector, std::vector< cv::Mat >& imagePointsVector1, std::vector< cv::Mat >& imagePointsVector2 )
{
    cv::Mat
        t1 = cv::Mat::zeros(cv::Size(3,1),cv::DataType<float>::type), 
        r1 = cv::Mat::zeros(cv::Size(3,1),cv::DataType<float>::type);
    
    cv::Vec3d 
        t2, r2;
    
    decomposeTransformation(*g_12_, r2, t2);
    
    cv::Ptr<cv::Mat>
        imagePoints1,
        imagePoints2;
        
    imagePointsVector1.clear();
    imagePointsVector2.clear();
    
    for (std::size_t actualNeighborIndex = 0; actualNeighborIndex < pointsGroupVector.size(); actualNeighborIndex++)
    {    
        imagePoints1 = new cv::Mat(cv::Size(1,1), CV_64FC2, cv::Scalar(0,0));
        imagePoints2 = new cv::Mat(cv::Size(1,1), CV_64FC2, cv::Scalar(0,0));
        
        cv::projectPoints(pointsGroupVector.at(actualNeighborIndex), r1, t1, *camera_matrix_, *distortion_coefficients_, *imagePoints1);
        cv::projectPoints(pointsGroupVector.at(actualNeighborIndex), r2, t2, *camera_matrix_, *distortion_coefficients_, *imagePoints2);
        
        imagePointsVector1.push_back(*imagePoints1);
        imagePointsVector2.push_back(*imagePoints2);
    }
}

void SingleCameraTriangulator::projectPointsToImages(const cv::Mat& pointsGroup, cv::Mat& imagePoints1, cv::Mat& imagePoints2)
{
    cv::Mat
        t1 = cv::Mat::zeros(cv::Size(3,1),cv::DataType<float>::type), 
        r1 = cv::Mat::zeros(cv::Size(3,1),cv::DataType<float>::type);
    
    cv::Vec3d 
        t2, r2;
    
    decomposeTransformation(*g_12_, r2, t2);
    
    cv::projectPoints(pointsGroup, r1, t1, *camera_matrix_, *distortion_coefficients_, imagePoints1);
    cv::projectPoints(pointsGroup, r2, t2, *camera_matrix_, *distortion_coefficients_, imagePoints2);
}


void SingleCameraTriangulator::projectPointsAndComputeResidual(const std::vector< cv::Mat >& pointsGroupVector, std::vector< cv::Mat >& imagePointsVector1, std::vector< cv::Mat >& imagePointsVector2, std::vector< std::vector< double > >& residualsVectors)
{
    imagePointsVector1.clear();
    imagePointsVector2.clear();
    
    projectPointsToImages(pointsGroupVector, imagePointsVector1, imagePointsVector2);
    
    // both points vectors should be of the same size TODO: check
    int
        numberOfGroups = imagePointsVector1.size();
            
    for (std::size_t i = 0; i < numberOfGroups; i++)
    {
//         std::cout << points.at(i) << std::endl;
//         std::cout << points.at(i).rows << " - " << points.at(i).cols << std::endl;
        int
            numberOfPoints = pointsGroupVector.at(i).cols;
        
        std::vector<double>
            groupResidual;
            
        for (std::size_t k = 0; k < numberOfPoints; k++)
        {
            cv::Point2d
                point1(imagePointsVector1.at(i).at<cv::Vec2d>(k)),
                point2(imagePointsVector2.at(i).at<cv::Vec2d>(k));
            
            double
                pixel1 = getBilinearInterpPix32f(*img_1_, point1.x, point1.y),
                pixel2 = getBilinearInterpPix32f(*img_2_, point2.x, point2.y);
                
            groupResidual.push_back((pixel1 - pixel2));
            
//             std::cout << point << " - " << pixel << " - " << colors.at(i)[0] << " " << colors.at(i)[1] << " " << colors.at(i)[2] << std::endl;
        }
        residualsVectors.push_back(groupResidual);
    }
}

void SingleCameraTriangulator::projectPointsAndComputeResidual(const cv::Mat& pointsGroup, cv::Mat& imagePoints1, cv::Mat& imagePoints2, std::vector< double >& residualsVector)
{
    projectPointsToImages(pointsGroup, imagePoints1, imagePoints2);
    
    int
        numberOfPoints = pointsGroup.cols;
        
    for (std::size_t k = 0; k < numberOfPoints; k++)
    {
        cv::Point2d
        point1(imagePoints1.at<cv::Vec2d>(k)),
        point2(imagePoints2.at<cv::Vec2d>(k));
        
        uchar
        pixel1 = img_1_->at<uchar>(round(point1.y),round(point1.x)),
        pixel2 = img_2_->at<uchar>(round(point2.y),round(point2.x));
        
        residualsVector.push_back((pixel1 - pixel2));
        
        //             std::cout << point << " - " << pixel << " - " << colors.at(i)[0] << " " << colors.at(i)[1] << " " << colors.at(i)[2] << std::endl;
    }
}

void SingleCameraTriangulator::extractPixelsContour(const cv::Vec2d &point, const double scale, std::vector< Pixel >& pixels)
{
    // TODO: prepare a lookuptable or something faster than this
    // Check all the pixel in the square centered at the interest point
    for(int i = -pixels_ray_; i <= pixels_ray_; i++)
    {
        for(int j = -pixels_ray_; j <= pixels_ray_; j++)
        {
            // If the pixel is inside the circle of ray: ray, take it
            if (i*i + j*j <= pixels_ray_*pixels_ray_)
            {
                Pixel p;
                p.x_ = point[0] + i;
                p.y_ = point[1] + j;
                p.i_ = getBilinearInterpPix32f(*img_1_, scale * p.x_, scale * p.y_);//img_1_->at<uchar>(point);
                
                pixels.push_back(p);
            }
        }
    }
}

void SingleCameraTriangulator::extractPixelsContour(const cv::Vec3d& point, const double scale, std::vector< Pixel >& pixels)
{
    // Go back to the pixel in image1
    cv::Mat
        pixelMat, undistortedPixelMat,
        pointMat = cv::Mat::zeros(cv::Size(3,1), CV_64FC1);
    
    // projectPoints needs a Mat :(
    pointMat.at<double>(0) = point[0];
    pointMat.at<double>(1) = point[1];
    pointMat.at<double>(2) = point[2];
    
    cv::projectPoints(pointMat,  cv::Mat::zeros(cv::Size(3,1),cv::DataType<float>::type), 
                      cv::Mat::zeros(cv::Size(3,1),cv::DataType<float>::type), *camera_matrix_, 
                      *distortion_coefficients_, pixelMat);
    
    cv::Vec2d 
        pixel = pixelMat.at<cv::Vec2d>(0);
    
    pixels.clear();
    extractPixelsContour(pixel, scale, pixels);
}

int SingleCameraTriangulator::getMdat()
{
    int returnValue = 0;
    
    // TODO: prepare a lookuptable or something faster than this
    // Check all the pixel in the square centered at the interest point
    for(int i = -pixels_ray_; i <= pixels_ray_; i++)
    {
        for(int j = -pixels_ray_; j <= pixels_ray_; j++)
        {
            // If the pixel is inside the circle of ray: ray, take it
            if (i*i + j*j <= pixels_ray_*pixels_ray_)
            {
                returnValue++;
            }
        }
    }
    
    return returnValue;
}


void SingleCameraTriangulator::projectPointToPlane(const cv::Vec3d& idealPoint, const cv::Vec3d &featurePoint, const cv::Vec3d& normal, cv::Vec3d &pointOnThePlane)
{    
    // The idealPoint is in the ideal camera coordinate with z=1
    // I have to find the intersection between the line generated by the point (treatened as a vector)
    // and the plane defined by the normal and the 3d feature point
    
    /**
     * v is the idealPoint treatened as a vector
     * p is the pointOnThePlane
     * n is the normal
     * p0 is the featurePoint
     * 
     * The system to solve is:
     * 
     * n(p-p0)=0 -> the plane equation
     * kv=p -> the vector mul by a value k reach the plane -> p is our point!
     * 
     * That give these equation to be solved:
     * 
     * k * v_x = p_x
     * k * v_y = p_y
     * k * v_z = p_z
     * 
     * k = m/n
     * 
     * m = n_x * p0_x + n_y * p0_y + n_z * p0_z
     * n = n_x * v_x + n_y * v_y + n_z * v_z
     */
    
    double m = ( normal[0]*featurePoint[0] + 
                 normal[1]*featurePoint[1] +
                 normal[2]*featurePoint[2] );
    
    double n = ( normal[0]*idealPoint[0] + 
                 normal[1]*idealPoint[1] +
                 normal[2]*idealPoint[2] ); 
    
    double k = m/n;
    
    
    pointOnThePlane[0] = k * idealPoint[0];
    pointOnThePlane[1] = k * idealPoint[1];
    pointOnThePlane[2] = k * idealPoint[2];
    
    if (isnan(pointOnThePlane[0]) || isnan(pointOnThePlane[1]) || isnan(pointOnThePlane[2])) 
    {
        std::cout << "male" << std::endl;
    }
}

void SingleCameraTriangulator::extractPixelsContourAndGet3DPoints(const cv::Vec3d& point, const cv::Vec3d& normal, std::vector< Pixel >& pixels, std::vector< cv::Vec3d >& pointsGroup)
{
    // Go back to the pixel in image1
    cv::Mat
        pixelMat, undistortedPixelMat,
        pointMat = cv::Mat::zeros(cv::Size(3,1), CV_64FC1);
        
    // projectPoints needs a Mat :(
    pointMat.at<double>(0) = point[0];
    pointMat.at<double>(1) = point[1];
    pointMat.at<double>(2) = point[2];
    
    cv::projectPoints(pointMat,  cv::Mat::zeros(cv::Size(3,1),cv::DataType<float>::type), 
                      cv::Mat::zeros(cv::Size(3,1),cv::DataType<float>::type), *camera_matrix_, 
                      *distortion_coefficients_, pixelMat);
    
    cv::Vec2d 
        pixel = pixelMat.at<cv::Vec2d>(0);
    
    pixels.clear();
    extractPixelsContour(pixel, pixels);
    
    // Recycle PixelMat
    pixelMat = cv::Mat::zeros(cv::Size(1, pixels.size()), CV_64FC2);
    for (std::size_t i = 0; i < pixels.size(); i++)
    {
        pixelMat.at<cv::Vec2d>(i) = cv::Vec2d(pixels.at(i).x_, pixels.at(i).y_);
    }
    
    cv::undistortPoints(pixelMat, undistortedPixelMat, *camera_matrix_, *distortion_coefficients_);
    
//     std::cout << undistortedPixelMat << std::endl;
    
    // For each pixel get a 3D point and store in the vector
    for (std::size_t i = 0; i < pixels.size(); i++)
    {
        cv::Vec2d 
            undistortedPixel = undistortedPixelMat.at<cv::Vec2d>(i);
        // The ideal point is in the ideal camera with z = 1
        cv::Vec3d
            idealPoint(undistortedPixel[0], undistortedPixel[1], 1),
            newPoint;
            
        projectPointToPlane(idealPoint, point, normal, newPoint);
        
        pointsGroup.push_back(newPoint);
    }
    
    cv::Mat
        test1, img1_BGR = cv::imread("test1.pgm", CV_LOAD_IMAGE_COLOR);
    
    drawBackProjectedPoints(img1_BGR, test1, pixelMat, cv::Scalar(255,0,0));
    
    cv::imwrite("test1.pgm", test1);

//     viewPointCloud(pointsGroup);
}

void SingleCameraTriangulator::get3dPointsFromImage1Pixels(const cv::Vec3d& point, const cv::Vec3d& normal, const std::vector< Pixel >& pixels, std::vector< cv::Vec3d >& pointsGroup)
{
    // Go back to the pixel in image1
    cv::Mat
        pixelMat, undistortedPixelMat;
    
    pixelMat = cv::Mat::zeros(cv::Size(1, pixels.size()), CV_64FC2);
    for (std::size_t i = 0; i < pixels.size(); i++)
    {
        pixelMat.at<cv::Vec2d>(i) = cv::Vec2d(pixels.at(i).x_, pixels.at(i).y_);
    }
    
    cv::undistortPoints(pixelMat, undistortedPixelMat, *camera_matrix_, *distortion_coefficients_);
    
    //     std::cout << undistortedPixelMat << std::endl;
    
    // For each pixel get a 3D point and store in the vector
    for (std::size_t i = 0; i < pixels.size(); i++)
    {
        cv::Vec2d 
            undistortedPixel = undistortedPixelMat.at<cv::Vec2d>(i);
        // The ideal point is in the ideal camera with z = 1
        cv::Vec3d
            idealPoint(undistortedPixel[0], undistortedPixel[1], 1),
            newPoint;
        
        projectPointToPlane(idealPoint, point, normal, newPoint);
        
        pointsGroup.push_back(newPoint);
    }
    
//     cv::Mat
//         test1, img1_BGR = cv::imread("test1.pgm", CV_LOAD_IMAGE_COLOR);
//     
//     drawBackProjectedPoints(img1_BGR, test1, pixelMat, cv::Scalar(255,0,0));
//     
//     cv::imwrite("test1.pgm", test1);
    
}

void SingleCameraTriangulator::updateImage1PixelsIntensity(const double scale, std::vector< Pixel >& pixels)
{
    for (std::vector<Pixel>::iterator it = pixels.begin(); it != pixels.end(); it++)
    {
        it->i_ = getBilinearInterpPix32f(*img_1_, scale * it->x_, scale * it->y_);
    }
}

void SingleCameraTriangulator::projectPointsToImage2(const std::vector< cv::Vec3d >& pointsGroup, const double scale, std::vector< Pixel >& pixels)
{
    cv::Vec3d 
        t2, r2;
    
    decomposeTransformation(*g_12_, r2, t2);
    
    cv::Mat imagePoints2;
    
    cv::projectPoints(pointsGroup, r2, t2, *camera_matrix_, *distortion_coefficients_, imagePoints2);
    
//     std::cout << imagePoints2 << std::endl;
    
    for (std::size_t i = 0; i < imagePoints2.rows; i++)
    {
        cv::Vec2d pixel = imagePoints2.at<cv::Vec2d>(i);
        Pixel p;
        p.x_ = pixel[0];
        p.y_ = pixel[1];
        p.i_ = getBilinearInterpPix32f(*img_2_, scale * p.x_, scale * p.y_);
        
        pixels.push_back(p);
    }
    
//     cv::Mat
//         test2, img2_BGR = cv::imread("test2.pgm", CV_LOAD_IMAGE_COLOR);
//     
//     drawBackProjectedPoints(img2_BGR, test2, imagePoints2, cv::Scalar(255,0,0));
//     
//     cv::imwrite("test2.pgm", test2);
    
//     cv::namedWindow("test2");
//     cv::imshow("test2", img2_BGR);
//     cv::waitKey();
}

