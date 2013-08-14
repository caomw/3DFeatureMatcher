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

#include "normaloptimizer.h"

#define THETA_INDEX 1
#define PHI_INDEX 0

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
        std::cout << "male" << std::endl;
        exit(-3);
    }
    
    std::vector<cv::Vec3d>
        pointGroup;
    std::vector<Pixel>
        imagePoints2;
    
    // obtain 3D points
    D->sct->get3dPointsFromImage1Pixels(*(D->point), normal, *(D->imagePoints1), pointGroup);
    int 
        good = 0;
    
    // update imagePoints1 to actual scale pixels intensity
    good = D->sct->updateImage1PixelsIntensity(D->scale, *(D->imagePoints1));
    
    if (good != 0)
    {
        (*info) = -1;
        return;
    }
    
    // get imagePoints2 at actual scale
    good = D->sct->projectPointsToImage2(pointGroup, D->scale, imagePoints2);
    
    if (good != 0)
    {
        (*info) = -1;
        return;
    }
    
//     if (1.0 == D->scale || 0.0625 == D->scale)
//     {
//         viewPointCloud(pointGroup, normal);
//         D->pvt->updateClouds(pointGroup, normal, *(D->color));
//     }
    
    for (std::size_t i = 0; i < m_dat; i++)
    {
        fvec[i] = (127 + abs(theta - D->thetaInitialGuess) + abs(phi - D->phiInitialGuess)) * (D->imagePoints1->at(i).i_ - imagePoints2.at(i).i_);
    }
}

NormalOptimizer::NormalOptimizer(const cv::FileStorage settings, SingleCameraTriangulator *sct)
{
    // Get the number of pyramid levels to compute
    settings["Neighborhoods"]["pyramids"] >> pyr_levels_;
    
    settings["Neighborhoods"]["epsilonLMMIN"] >> epsilon_lmmin_;
    
    sct_ = sct;
    
    visualizer_ = new pclVisualizerThread();
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
        data = { sct_, actual_point_, m_dat_, actual_scale_, &image_1_points_, visualizer_, color_, theta, phi };
        
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
    
    if (status.info < 0)
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


void NormalOptimizer::computeOptimizedNormals(std::vector<cv::Vec3d> &points3D, std::vector< cv::Vec3d >& normalsVector, std::vector<cv::Scalar> &colors)
{
    //Start visualizer thread
    boost::thread workerThread(*visualizer_); 
    
    int index = 0;
    for (std::vector<cv::Vec3d>::iterator actualPointIT = points3D.begin(); actualPointIT != points3D.end(); actualPointIT++)
    {
        /*DEBUG*/
        std::cout << "Punto su cui sto lavorando: " << index << std::endl;
        /*DEBUG*/
        
        // get the point and compute the initial guess for the normal
        actual_point_ = new cv::Vec3d((*actualPointIT));
        actual_norm_ = new cv::Vec3d((*actual_point_) / cv::norm(*actual_point_));

        // Get the neighborhood of the feature point pixel
        sct_->extractPixelsContour((*actual_point_), image_1_points_);
        
        // Set mdat for actual point
        m_dat_ = image_1_points_.size();
        
        // Set the color for the visualizer
        color_ = new cv::Scalar(colors[index++]);
        
        if ( m_dat_ > 0)
        {
            if (!optimize_pyramid())
            {
                // remove the point, it is bad
                // points3D.erase(actualPointIT);
                // remove it later
                std::cout << "Bad point!!" << std::endl;
            }
            else
            {
                visualizer_->keepLastCloud();
                normalsVector.push_back((*actual_norm_));
            }
        }
    }
    
    // join the thread
    workerThread.join(); 
    
    std::cout << points3D.size() << " - " << normalsVector.size() << std::endl;
}

