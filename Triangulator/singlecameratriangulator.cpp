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
    
}

void SingleCameraTriangulator::setg12(const cv::Vec3d& T1, const cv::Vec3d& T2, const cv::Vec3d& rodrigues1, const cv::Vec3d& rodrigues2)
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
