/**
 * \file normaloptimizer.cpp
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

///TODO: remove elevation(theta) parameter and use only the azimuth

#include "normaloptimizer.h"

#define THETA_INDEX 1
#define PHI_INDEX 0

#ifndef ENABLE_VISUALIZER_
#define ENABLE_VISUALIZER_
#endif

typedef struct {
    
    SingleCameraTriangulator 
        *sct;
    cv::Vec3d 
        *point;
    int 
        m_dat;
    double 
        scale;
    std::vector<Pixel> 
        *imagePoints1;
    cv::Mat
        *imagePoints1_MAT;
    pclVisualizerThread
        *pvt;
    cv::Scalar
        *color;
        
    double
        thetaInitialGuess,
        phiInitialGuess;
    
} lmminDataStruct;

// Function that evaluate fvec of lmmin
void evaluateNormal( const double *par, int m_dat,
                     const void *data, double *fvec,
                     int *info )
{
    lmminDataStruct 
        *D = (lmminDataStruct*) data;
    cv::Vec3d 
        normal;
    
    double
        phi = par[PHI_INDEX],
        theta = par[THETA_INDEX];
    
    sph2car(phi, theta, normal);
    
    // lmmin can go over 1 for normal coordinate, which is wrong for a normal versor.
    if (isnan(normal[2]) || isnan(normal[1]) || isnan(normal[0]))
    {
        (*info) = -1;
        return;
    }
    
    std::vector<cv::Vec3d>
        pointGroup;
    std::vector<Pixel>
        imagePoints2;
    int 
        goodNormal = 0;
    
    // obtain 3D points
    goodNormal += D->sct->get3dPointsFromImage1Pixels(*(D->point), normal, *(D->imagePoints1_MAT), pointGroup);
    
    if (0 != goodNormal)
    {
        (*info) = -1;
        return;
    }

    // update imagePoints1 to actual scale pixels intensity
    goodNormal += D->sct->updateImage1PixelsIntensity(D->scale, *(D->imagePoints1));
    
    if (0 != goodNormal)
    {
        (*info) = -1;
        return;
    }
    
    // get imagePoints2 at actual scale
    goodNormal += D->sct->projectPointsToImage2(pointGroup, D->scale, imagePoints2);
    
    if (0 != goodNormal)
    {
        (*info) = -1;
        return;
    }
    
#ifdef ENABLE_VISUALIZER_
    D->pvt->updateClouds(pointGroup, normal, *(D->color));
#endif
    
    // Compute weight
    double 
        w_theta = 1.0,
        w_phi = 1.0,
        w;
        
    // theta must lie in [-M_PI, M_PI]
    // phi must lie in [-M_PI/2, M_PI/2]
    if (abs(theta) - M_PI/2 > 0 || abs(phi) - M_PI > 0)
    {
        w_theta = exp(abs(theta) - M_PI/2) + 1;
        std::cout << "--" << w_theta;
        
        w_phi = exp(abs(phi) - M_PI + 1) + 1;
        std::cout << "," << w_phi;
    }
    
    w = w_phi * w_theta;
    
    // compute residuals
    for (std::size_t i = 0; i < m_dat; i++)
    {
        fvec[i] = w * (D->imagePoints1->at(i).i_ - imagePoints2.at(i).i_);
    }
}

NormalOptimizer::NormalOptimizer(const cv::FileStorage settings, SingleCameraTriangulator *sct)
{
    // Get the number of pyramid levels to compute
    settings["Neighborhoods"]["pyramids"] >> pyr_levels_;
    
    settings["Neighborhoods"]["epsilonLMMIN"] >> epsilon_lmmin_;
    
    sct_ = sct;
    
    // 2 Camera-IMU rotation
    std::vector<double>
        rodriguesIC_vector;
    
        settings["CameraSettings"]["rodriguesIC"] >> rodriguesIC_vector;
        
    cv::Vec3d
        rodriguesIC(rodriguesIC_vector[0],rodriguesIC_vector[1],rodriguesIC_vector[2]);
        
    cv::Matx33d rotation_IC;
    
    cv::Rodrigues(rodriguesIC, rotation_IC);
    
    // TODO: gravity set to y axes, check if it is correct
    cv::Vec3d temp(0,0,-1);
    temp = rotation_IC.inv() * (temp);
    gravity_ = new const cv::Vec3d(temp);
    
    std::cout << "Gravity: " << *gravity_ << std::endl;
    
#ifdef ENABLE_VISUALIZER_
    visualizer_ = new pclVisualizerThread();
#endif
}

cv::Vec3d NormalOptimizer::getGravity()
{
    return *gravity_;
}


void NormalOptimizer::setImages(const cv::Mat& img1, const cv::Mat& img2)
{
    // check if sct_ is null
    if (sct_ == 0)
    {
        exit(-1);
    }
    
    // compute pyrDown on images and save results in vectors
    compute_pyramids(img1, img2);
    
    // set images to sct to get the mdat
    sct_->setImages(img1, img2);
}

void NormalOptimizer::compute_pyramids(const cv::Mat& img1, const cv::Mat& img2)
{
    cv::Mat 
    pyr1 = img1, pyr2 = img2;
    
    pyr_img_1_.push_back(pyr1);
    pyr_img_2_.push_back(pyr2);
    
    for( int i = 1; i <= pyr_levels_; i++)
    {
        cv::pyrDown(pyr_img_1_[i-1], pyr1);
        cv::pyrDown(pyr_img_2_[i-1], pyr2);
        pyr_img_1_.push_back(pyr1);
        pyr_img_2_.push_back(pyr2);
    }
}

bool NormalOptimizer::optimize_pyramid()
{
    float 
        img_scale = float( pow(2.0,double(pyr_levels_)) );
        
    for( int i = pyr_levels_; i >= 0; i--)
    {
        actual_scale_ = 1.0/img_scale;
//         std::cout << "scala: " << actual_scale_ << " Actual normal: " << *actual_norm_;
        
        if (!optimize(i))
        {
            return false;
            std::cout << "BAD NORMAL - 3" << std::endl;
        }
        
//         std::cout << " Estimated normal: " << *actual_norm_ << std::endl;
        
        img_scale /= 2.0f;
    }
    
    return true;
}

bool NormalOptimizer::optimize(const int pyrLevel)
{
    // convert the normal to spherical coordinates
    double 
        theta, phi;
    
    car2sph((*actual_norm_), phi, theta);
    
    /* parameter vector */
    int 
        n_par = 2;  // number of parameters in evaluateNormal
    double 
        par[2];
    par[PHI_INDEX] = phi;
    par[THETA_INDEX] = theta;
        
    sct_->setImages(pyr_img_1_[pyrLevel],pyr_img_2_[pyrLevel]);
    
    lmminDataStruct
        data = { sct_, actual_point_, m_dat_, actual_scale_, &image_1_points_, image_1_points_MAT_, visualizer_, color_, theta, phi };
        
    /* auxiliary parameters */
    lm_status_struct 
        status;
    
    lm_control_struct 
        control = lm_control_double;
    control.epsilon = epsilon_lmmin_;
    
    lm_princon_struct 
        princon = lm_princon_std;
    princon.flags = 0;

    lmmin( n_par, par, m_dat_, &data, evaluateNormal,
           lm_printout_std, &control, 0/*&princon*/, &status );
    
    /// status.info == 11 -> interrupt for bad points
    if (status.info == 11)
    {
        return false;
    }
    
    sph2car(par[PHI_INDEX], par[THETA_INDEX], (*actual_norm_));
    
    return true;
}

void NormalOptimizer::computeOptimizedNormals(std::vector< cv::Vec3d >& points3D, std::vector< cv::Vec3d >& normalsVector)
{
    std::vector<cv::Scalar> colors;
    for (std::vector<cv::Vec3d>::iterator actualPointIT = points3D.begin(); actualPointIT != points3D.end(); actualPointIT++)
    {
        colors.push_back(cv::Scalar(150,150,255));
    }
    computeOptimizedNormals(points3D, normalsVector, colors);
}

void NormalOptimizer::startVisualizerThread()
{
#ifdef ENABLE_VISUALIZER_
    //Start visualizer thread
    workerThread_ = new boost::thread(*visualizer_); 
#endif
}

void NormalOptimizer::stopVisualizerThread()
{
#ifdef ENABLE_VISUALIZER_
    // join the thread
    workerThread_->join(); 
#endif
}


void NormalOptimizer::computeOptimizedNormals(std::vector<cv::Vec3d> &points3D, std::vector< cv::Vec3d >& normalsVector, std::vector<cv::Scalar> &colors)
{
    
    /// TODO: Move these images as class parameters -> use pyr[0]
//     cv::Mat 
//         img1_patches = cv::imread("/home/mpp/WorkspaceTesi/loop_dataset/Images/img_0000000750.pgm", CV_LOAD_IMAGE_COLOR);
//     cv::Mat 
//         img2_patches = cv::imread("/home/mpp/WorkspaceTesi/loop_dataset/Images/img_0000000770.pgm", CV_LOAD_IMAGE_COLOR);
    
    cv::Mat
        img1_patches = pyr_img_1_[0].clone(),
        img2_patches = pyr_img_2_[0].clone();
    
    int index = 0;
    for (std::vector<cv::Vec3d>::iterator actualPointIT = points3D.begin(); actualPointIT != points3D.end();/* actualPointIT++*/)
    {
        /*DEBUG*/
        std::cout << "Punto su cui sto lavorando: " << index << std::endl;
        /*DEBUG*/
        
        // get the point and compute the initial guess for the normal
        actual_point_ = new cv::Vec3d((*actualPointIT));
        actual_norm_ = new cv::Vec3d((*actual_point_) / cv::norm(*actual_point_));
        
        double theta, phi, angle;
        angle = atan2((*actual_norm_)[2], (*actual_norm_)[1]) * 180 / M_PI;
        car2sph(*actual_norm_, phi, theta);
        
//         std::cout << *actual_norm_ << " - [" << phi * 180 / M_PI << ", " << theta * 180 / M_PI << "] - (" << angle << ")" << std::endl;

        // Get the neighborhood of the feature point pixel
        sct_->extractPixelsContour((*actual_point_), image_1_points_);
        
        image_1_points_MAT_ = new cv::Mat(cv::Size(1, image_1_points_.size()), CV_64FC2, cv::Scalar(0));
        for (std::size_t i = 0; i < image_1_points_.size(); i++)
        {
            image_1_points_MAT_->at<cv::Vec2d>(i) = cv::Vec2d(image_1_points_.at(i).x_, image_1_points_.at(i).y_);
        }
        
        // Set mdat for actual point
        m_dat_ = image_1_points_.size();
        
        
        if ( 0 >= m_dat_ )
        {
            points3D.erase(actualPointIT);
            std::cout << "Not enough pixels!" << std::endl;
            continue;
        }
        
        // Set the color for the visualizer
        color_ = new cv::Scalar(colors[index++]);
        
        if ( angle < 105 )
        {
            if (!optimize_pyramid())
            {
                points3D.erase(actualPointIT);
                std::cout << "Bad point!" << std::endl;
                
                continue;
            }
#ifdef ENABLE_VISUALIZER_
            visualizer_->keepLastCloud();
#endif
        }
        else
        {
            (*actual_norm_)[0] = (*gravity_)[0];
            (*actual_norm_)[1] = (*gravity_)[1];
            (*actual_norm_)[2] = (*gravity_)[2];
            
            std::vector<cv::Vec3d>
                pointGroup;
            sct_->get3dPointsFromImage1Pixels(*actual_point_, *actual_norm_, *image_1_points_MAT_, pointGroup);
#ifdef ENABLE_VISUALIZER_
            visualizer_->updateClouds(pointGroup, *actual_norm_, *color_);
            visualizer_->keepLastCloud();
#endif
        }
        
        /// Here I can draw images

//                 //// CODE TO DRAW image 1
//                 for( std::vector<Pixel>::iterator it = image_1_points_.begin(); it != image_1_points_.end(); it++)
//                 {
// //                     cv::Vec3b pixel_color(color_[0], color_[1], color_[2]);
//                     
//                     cv::Point2d
//                         point ((*it).x_, (*it).y_);
//                     
//                     cv::Point2i
//                         pixel(round(point.x),round(point.y));
//                         
//                     img1_patches.at<cv::Vec3b>(pixel)[0] = (*color_)[0];
//                     img1_patches.at<cv::Vec3b>(pixel)[1] = (*color_)[1];
//                     img1_patches.at<cv::Vec3b>(pixel)[2] = (*color_)[2];
//                 }
//                 cv::imwrite("image1pixels.pgm", img1_patches);

//                 //// CODE TO DRAW image 2
//                 std::vector<cv::Vec3d>
//                     pointGroup;
//                 std::vector<Pixel>
//                     imagePoints2;
//                 
//                 // obtain 3D points
//                 sct_->get3dPointsFromImage1Pixels(*actual_point_, *actual_norm_, *image_1_points_MAT_, pointGroup);
//                 
//                 sct_->projectPointsToImage2(pointGroup, 1.0, imagePoints2);
//                 for( std::vector<Pixel>::iterator it = imagePoints2.begin(); it != imagePoints2.end(); it++)
//                 {
// //                     cv::Vec3b pixel_color(color_[0], color_[1], color_[2]);
//                     
//                     cv::Point2d
//                         point ((*it).x_, (*it).y_);
//                     
//                     cv::Point2i
//                         pixel(round(point.x),round(point.y));
//                         
//                     img2_patches.at<cv::Vec3b>(pixel)[0] = (*color_)[0];
//                     img2_patches.at<cv::Vec3b>(pixel)[1] = (*color_)[1];
//                     img2_patches.at<cv::Vec3b>(pixel)[2] = (*color_)[2];
//                 }
//                 cv::imwrite("image2pixels.pgm", img2_patches);
        
        normalsVector.push_back((*actual_norm_));
        actualPointIT++;
    }
    
    std::cout << points3D.size() << " - " << normalsVector.size() << std::endl;
}

void NormalOptimizer::computeFeaturesFrames(std::vector< cv::Vec3d >& points3D, std::vector< cv::Vec3d >& normalsVector, std::vector< cv::Matx44d >& featuresFrames)
{
    cv::Matx44d
        actualFrame;
        
    cv::Vec3d
        x, 
        y, 
        z,
        e1(1,0,0), 
        e2(0,1,0), 
        e3(0,0,1);
    
    std::vector<cv::Vec3d>::iterator 
        pt = points3D.begin(),
        nr = normalsVector.begin();
        
    while ( pt != points3D.end() && nr != normalsVector.end() )
    {
        // z is set equal to the normal
        z = (*nr);
        // x is perpendicular to the plane defined by the gravity and z
        x = gravity_->cross(z) /*/ cv::norm(gravity_->cross(z))*/;
        // y is perpendicular to the plane z-x
        y = z.cross(x) /*/ cv::norm(z.cross(x))*/;
        
        cv::normalize(x,x);
        cv::normalize(y,y);
        
        // put the basis as columns in the matrix
        actualFrame(0,0) = e1.dot(x); actualFrame(0,1) = e1.dot(y); actualFrame(0,2) = e1.dot(z); actualFrame(0,3) = (*pt)[0];
        actualFrame(1,0) = e2.dot(x); actualFrame(1,1) = e2.dot(y); actualFrame(1,2) = e2.dot(z); actualFrame(1,3) = (*pt)[1];
        actualFrame(2,0) = e3.dot(x); actualFrame(2,1) = e3.dot(y); actualFrame(2,2) = e3.dot(z); actualFrame(2,3) = (*pt)[2];
        actualFrame(3,0) = 0;         actualFrame(3,1) = 0;         actualFrame(3,2) = 0;         actualFrame(3,3) = 1;
        
        
        featuresFrames.push_back(actualFrame);
        
//         std::cout << x.dot(y) << " - " << x.dot(z) << " - " << y.dot(z) << std::endl;
        
//         cv::Matx33d rot;
//         
//         rot(0,0) = actualFrame(0,0); rot(0,1) = actualFrame(0,1); rot(0,2) = actualFrame(0,2);
//         rot(1,0) = actualFrame(1,0); rot(1,1) = actualFrame(1,1); rot(1,2) = actualFrame(1,2);
//         rot(2,0) = actualFrame(2,0); rot(2,1) = actualFrame(2,1); rot(2,2) = actualFrame(2,2);
        
//         std::cout << rot.t() << " --- " << std::endl << rot.inv() << std::endl;
        
        pt++; nr++;
    }
    
}