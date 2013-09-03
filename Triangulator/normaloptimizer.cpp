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

#include <limits>

#define THETA_INDEX 1
#define PHI_INDEX 0

#ifndef ENABLE_VISUALIZER_
#define ENABLE_VISUALIZER_
#endif

void manageBadData(int *fvecIndex, int m_dat, double *fvec)
{
    // compute residuals
    int 
        start = *fvecIndex, 
        end = *fvecIndex + m_dat - 1;
    for (std::size_t i = start; i < end; i++)
    {
        fvec[i] = 0; // max difference from black (0) and white (255)
    }
    *fvecIndex = *fvecIndex + m_dat;
}

// Function that evaluate fvec of lmmin
void evaluateNormal( const double *par, int m_dat,
                     const void *data, double *fvec,
                     int *info )
{
    lmminDataStruct 
        *dataStruct = (lmminDataStruct *) data;
        
    int 
        index,
        gOffset = 6,
        fvecIndex = 0;
        
    std::vector< std::vector<cv::Vec3d> > 
        pointGroupVector;
        
    cv::Vec3d rodrigues, T;
    
    rodrigues[0] = par[0];
    rodrigues[1] = par[1];
    rodrigues[2] = par[2];
    
    T[0] = par[3];
    T[1] = par[4];
    T[2] = par[5];
    
    for (int i = 0, index = gOffset; i < dataStruct->vectorSize; i++, index += 3)
    {
        if (!(dataStruct->isGood->at(i)))
        {
            // Put zero residuals
            manageBadData(&fvecIndex, dataStruct->m_dat->at(i), fvec);
            pointGroupVector.push_back(std::vector<cv::Vec3d>());
            continue;
        }
        
        double
            depth = par[index + 0],
            phi = par[index + 1],
            theta = par[index + 2];
        
        sph2car(phi, theta, dataStruct->normal->at(i));
        
        std::vector<cv::Vec3d>
            pointGroup;
        std::vector<Pixel>
            imagePoints2;
        int 
            goodNormal = 0;
        
//         double w_depth = exp(cv::norm(dataStruct->point->at(i) - dataStruct->point->at(i) * exp(depth)) + 1);
            
        cv::Vec3d optimizedPoint = dataStruct->point->at(i) * exp(depth);
            
        // obtain 3D points
        dataStruct->sct->setg12(T, rodrigues);
        goodNormal += dataStruct->sct->get3dPointsFromImage1Pixels(optimizedPoint, dataStruct->normal->at(i),
                                                                   dataStruct->imagePoints1_MAT->at(i), pointGroup);
        
        if (0 != goodNormal)
        {
            dataStruct->isGood->at(i) = false;
            // Put zero residuals
            manageBadData(&fvecIndex, dataStruct->m_dat->at(i), fvec);
            pointGroupVector.push_back(std::vector<cv::Vec3d>());
            continue;
        }
        
        // update imagePoints1 to actual scale pixels intensity
        goodNormal += dataStruct->sct->updateImage1PixelsIntensity(dataStruct->scale, dataStruct->imagePoints1->at(i));
        
        if (0 != goodNormal)
        {
            dataStruct->isGood->at(i) = false;
            // Put zero residuals
            manageBadData(&fvecIndex, dataStruct->m_dat->at(i), fvec);
            pointGroupVector.push_back(std::vector<cv::Vec3d>());
            continue;
        }
        
        // get imagePoints2 at actual scale
        goodNormal += dataStruct->sct->projectPointsToImage2(pointGroup, dataStruct->scale, imagePoints2);
        
        if (0 != goodNormal)
        {
            dataStruct->isGood->at(i) = false;
            // Put zero residuals
            manageBadData(&fvecIndex, dataStruct->m_dat->at(i), fvec);
            pointGroupVector.push_back(std::vector<cv::Vec3d>());
            continue;
        }
        
#ifdef ENABLE_VISUALIZER_
        // For viz
        pointGroupVector.push_back(pointGroup);
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
        for (std::size_t h = 0; h < dataStruct->m_dat->at(i); h++)
        {
            fvec[fvecIndex++] = w * (dataStruct->imagePoints1->at(i).at(h).i_ - imagePoints2.at(h).i_);
        }
    }
    
#ifdef ENABLE_VISUALIZER_
    dataStruct->pvt->updateClouds(pointGroupVector, *(dataStruct->normal), *(dataStruct->color));
#endif
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
    
    double az, el;
    
    car2sph(*gravity_, az, el);
    
    std::cout << "Gravity: " << *gravity_ << "; [" << az << "," << el << "]" << std::endl;
    
    fixed_angle_ = az + M_PI / 2;
    
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

void NormalOptimizer::computeOptimizedNormalsAllInOne(std::vector< cv::Vec3d >& points3D, std::vector< cv::Vec3d >& normalsVector, std::vector< cv::Scalar >& colors)
{
    // Prepare data for computation
        
    std::vector< cv::Mat >
        image1MatVector;
        
    std::vector< std::vector<Pixel> >
        image1PixelVector;
        
    std::vector< cv::Vec3d >
        tempNormalsVector;
        
    std::vector< bool >
        goodVector;
        
    std::vector< int >
        m_datVector;
    int global_m_dat = 0;
        
    int index = 0;
    for (std::vector<cv::Vec3d>::iterator actualPointIT = points3D.begin(); actualPointIT != points3D.end();)
    {
        cv::Vec3d  
            normal((*actualPointIT) / cv::norm(*actualPointIT));
        
        // Get the neighborhood of the feature point pixel
        std::vector<Pixel>
            image1Pixels;
        sct_->extractPixelsContour((*actualPointIT), image1Pixels);
        
        cv::Mat
            image1PointsMAT(cv::Size(1, image1Pixels.size()), CV_64FC2, cv::Scalar(0));
        for (std::size_t i = 0; i < image1Pixels.size(); i++)
        {
            image1PointsMAT.at<cv::Vec2d>(i) = cv::Vec2d(image1Pixels.at(i).x_, image1Pixels.at(i).y_);
        }
        
        // Set mdat for actual point
        int m_dat = image1Pixels.size();
        
        if ( 0 >= m_dat )
        {
            points3D.erase(actualPointIT);
            std::cout << "Not enough pixels!" << std::endl;
            continue;
        }
        
        tempNormalsVector.push_back(normal);
        image1PixelVector.push_back(image1Pixels);
        image1MatVector.push_back(image1PointsMAT.clone());
        m_datVector.push_back(m_dat);
        goodVector.push_back(true);
    
        index++;
        actualPointIT++;
    }
    
    lmminDataStruct
        optimizationDataVector = { sct_, &points3D, &tempNormalsVector, 
        global_m_dat, &m_datVector, 0.0, &image1PixelVector, 
        &image1MatVector, visualizer_, &colors, &goodVector, points3D.size() };
        
    optimize_pyramid_all(optimizationDataVector);
    
    std::vector< cv::Vec3d >::iterator normalIT = tempNormalsVector.begin();
    std::vector< cv::Vec3d >::iterator pointIT = points3D.begin();
    
    int i = 0;
    while ( normalIT != tempNormalsVector.end(), pointIT != points3D.end() )
    {
        if (!(optimizationDataVector.isGood->at(i++)))
        {
            points3D.erase(pointIT);
            tempNormalsVector.erase(normalIT);
        }
        else
        {
            normalsVector.push_back(*normalIT);
            
            normalIT++;
            pointIT++;
        }
    }
    
//     std::cout << "Punti: " << points3D.size() << " - Normali trovate: " << normalsVector.size() << std::endl;
}

void NormalOptimizer::optimize_pyramid_all(lmminDataStruct &optimizationDataStruct)
{
    float 
        img_scale = float( pow(2.0,double(pyr_levels_)) );
    
    for( int i = pyr_levels_; i >= 0; i--)
    {
        actual_scale_ = 1.0/img_scale;
        
        std::cout << "Started the opt of a pyr layer, scale: " << actual_scale_ << std::endl;
        optimize_all(i, optimizationDataStruct);
        std::cout << std::endl << "End of the opt of the layer" << std::endl;
        
        img_scale /= 2.0f;
    }
}

void NormalOptimizer::optimize_all(const int pyrLevel, lmminDataStruct &optimizationDataStruct)
{
    int
        parPerPoint = 3,
        gOffset = 6;
    int
        n_par = gOffset + parPerPoint * optimizationDataStruct.point->size();
    
    double 
        par[n_par];
        
    cv::Vec3d rodrigues, T;
    
    sct_->getg12Params(T, rodrigues);
    
    par[0] = rodrigues[0];
    par[1] = rodrigues[1];
    par[2] = rodrigues[2];
    
    par[3] = T[0];
    par[4] = T[1];
    par[5] = T[2];    
    
    m_dat_ = 0;
        
    int 
        index = gOffset;
    
    for (int i = 0; i < optimizationDataStruct.vectorSize; i++)
    {
        par[index + 0] = 0.0;
        car2sph(optimizationDataStruct.normal->at(i), par[index + 1], par[index + 2]);
        
        //         std::cout << *((*dataIT).normal) << " - [" << par[index] << ", " << par[index + 1] << "]" << std::endl;
        
        optimizationDataStruct.scale = actual_scale_;
        
        m_dat_ = m_dat_ + optimizationDataStruct.m_dat->at(i);
        
        index = index + parPerPoint;
    }

    sct_->setImages(pyr_img_1_[pyrLevel],pyr_img_2_[pyrLevel]);
    
    /* auxiliary parameters */
    lm_status_struct 
    status;
    
    lm_control_struct 
        control = lm_control_double;
    control.epsilon = epsilon_lmmin_;
//     control.patience = 20;
    
    lm_princon_struct 
        princon = lm_princon_std;
    princon.flags = 0;
    
    lmmin( n_par, par, m_dat_, &optimizationDataStruct, evaluateNormal,
           lm_printout_std, &control, 0/*&princon*/, &status );
    
    index = gOffset;
    
    for (int i = 0; i < optimizationDataStruct.vectorSize; i++)
    {
        double 
            depth = par[index + 0];
        
        optimizationDataStruct.point->at(i) = optimizationDataStruct.point->at(i) * exp(depth);
            
        sph2car(par[index + 1], par[index + 2], optimizationDataStruct.normal->at(i));
        
        index = index + parPerPoint;
    }
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
        
        if (*gravity_ != z)
        {
            // x is perpendicular to the plane defined by the gravity and z
            x = gravity_->cross(z) /*/ cv::norm(gravity_->cross(z))*/;
        }
        else
        {
            x = cv::Vec3d(0,0,1).cross(z);
        }
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
