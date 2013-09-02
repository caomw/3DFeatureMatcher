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

void manageBadData(int *fvecIndex, lmminDataStruct *data, double *fvec)
{
    // compute residuals
    int 
        start = *fvecIndex, 
        end = *fvecIndex + data->m_dat - 1;
    for (std::size_t i = start; i < end; i++)
    {
        fvec[i] = 0; // max difference from black (0) and white (255)
    }
    *fvecIndex = *fvecIndex + data->m_dat;
}

// Function that evaluate fvec of lmmin
void evaluateNormal( const double *par, int m_dat,
                     const void *data, double *fvec,
                     int *info )
{
    std::vector<lmminDataStruct> 
        *dataVector = (std::vector<lmminDataStruct> *) data;
        
    std::vector<lmminDataStruct>::iterator 
        dataIT;
    int 
        index,
        fvecIndex = 0;
        
#ifdef ENABLE_VISUALIZER_
    std::vector< std::vector<cv::Vec3d> > 
        pointGroupVector;
    std::vector<cv::Vec3d> 
        normalVector;
    std::vector<cv::Scalar> 
        colorVector;
#endif
        
    for (dataIT = dataVector->begin(), index = 0; dataIT != dataVector->end(); dataIT++, index += 2)
    {
        if (!(dataIT->isGood))
        {
            // Put zero residuals
            manageBadData(&fvecIndex, &*dataIT, fvec);
            colorVector.push_back(cv::Scalar(0,0,0));
            normalVector.push_back(cv::Vec3d(0,0,0));
            pointGroupVector.push_back(std::vector<cv::Vec3d>());
            continue;
        }
        
        double
            phi = par[index],
            theta = par[index + 1];
            
        cv::Vec3d
            normal;
        
        sph2car(phi, theta, normal);
        
        std::vector<cv::Vec3d>
            pointGroup;
        std::vector<Pixel>
            imagePoints2;
        int 
            goodNormal = 0;
        
        // obtain 3D points
        goodNormal += dataIT->sct->get3dPointsFromImage1Pixels(*(dataIT->point), normal, *(dataIT->imagePoints1_MAT), pointGroup);
        
//         #ifdef ENABLE_VISUALIZER_
//         dataIT->pvt->updateClouds(pointGroup, normal, *(dataIT->color));
//         #endif
        
        if (0 != goodNormal)
        {
            dataIT->isGood = false;
            // Put zero residuals
            manageBadData(&fvecIndex, &*dataIT, fvec);
            colorVector.push_back(cv::Scalar(0,0,0));
            normalVector.push_back(cv::Vec3d(0,0,0));
            pointGroupVector.push_back(std::vector<cv::Vec3d>());
            continue;
        }
        
        // update imagePoints1 to actual scale pixels intensity
        goodNormal += dataIT->sct->updateImage1PixelsIntensity(dataIT->scale, *(dataIT->imagePoints1));
        
        if (0 != goodNormal)
        {
            dataIT->isGood = false;
            // Put zero residuals
            manageBadData(&fvecIndex, &*dataIT, fvec);
            colorVector.push_back(cv::Scalar(0,0,0));
            normalVector.push_back(cv::Vec3d(0,0,0));
            pointGroupVector.push_back(std::vector<cv::Vec3d>());
            continue;
        }
        
        // get imagePoints2 at actual scale
        goodNormal += dataIT->sct->projectPointsToImage2(pointGroup, dataIT->scale, imagePoints2);
        
        if (0 != goodNormal)
        {
            dataIT->isGood = false;
            // Put zero residuals
            manageBadData(&fvecIndex, &*dataIT, fvec);
            colorVector.push_back(cv::Scalar(0,0,0));
            normalVector.push_back(cv::Vec3d(0,0,0));
            pointGroupVector.push_back(std::vector<cv::Vec3d>());
            continue;
        }
        
#ifdef ENABLE_VISUALIZER_
        // For viz
        colorVector.push_back(*(dataIT->color));
        normalVector.push_back(normal);
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
        for (std::size_t i = 0; i < dataIT->m_dat; i++)
        {
            fvec[fvecIndex++] = w * (dataIT->imagePoints1->at(i).i_ - imagePoints2.at(i).i_);
        }
    }
    
#ifdef ENABLE_VISUALIZER_
    dataIT = dataVector->begin();
    dataIT->pvt->updateClouds(pointGroupVector, normalVector, colorVector);
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
    std::vector<lmminDataStruct>
        optimizationDataVector;
        
    std::vector< cv::Mat * >
        image1MatVector;
        
    std::vector< std::vector<Pixel> * >
        image1PixelVector;
        
    std::vector< cv::Vec3d * >
        tempNormalsVector;
        
    int index = 0;
    for (std::vector<cv::Vec3d>::iterator actualPointIT = points3D.begin(); actualPointIT != points3D.end();/* actualPointIT++*/)
    {
        cv::Vec3d  
            normal((*actualPointIT) / cv::norm(*actualPointIT));
        tempNormalsVector.push_back(new cv::Vec3d(normal));
        
        // Get the neighborhood of the feature point pixel
        std::vector<Pixel>
            image1Pixels;
        sct_->extractPixelsContour((*actualPointIT), image1Pixels);
        image1PixelVector.push_back( new std::vector<Pixel>(image1Pixels) );
        
        cv::Mat
            image1PointsMAT(cv::Size(1, image1Pixels.size()), CV_64FC2, cv::Scalar(0));
        for (std::size_t i = 0; i < image1Pixels.size(); i++)
        {
            image1PointsMAT.at<cv::Vec2d>(i) = cv::Vec2d(image1Pixels.at(i).x_, image1Pixels.at(i).y_);
        }
        image1MatVector.push_back(new cv::Mat(image1PointsMAT));
        
        // Set mdat for actual point
        int m_dat = image1Pixels.size();
        
        if ( 0 >= m_dat )
        {
            points3D.erase(actualPointIT);
            std::cout << "Not enough pixels!" << std::endl;
            continue;
        }
        
        lmminDataStruct 
            actualPointData = { sct_, &*actualPointIT, tempNormalsVector[index], 
                                m_dat, 0.0, image1PixelVector[index], 
                                image1MatVector[index], visualizer_, &(colors[index]), true };
        
                                
        std::cout << *(tempNormalsVector[index]) << std::endl;
                                
        optimizationDataVector.push_back(actualPointData);
    
        index++;
        actualPointIT++;
    }
    
    optimize_pyramid_all(optimizationDataVector);
    
    for (std::size_t t = 0; t < tempNormalsVector.size(); t++)
    {
        normalsVector.push_back(*(tempNormalsVector[t]));
    }
    
//     std::cout << "Punti: " << points3D.size() << " - Normali trovate: " << normalsVector.size() << std::endl;
    
    for (int i = 0; i < points3D.size(); i++)
    {
        delete(image1MatVector[i]);
        delete(image1PixelVector[i]);
        delete(tempNormalsVector[i]);
    }
}

void NormalOptimizer::optimize_pyramid_all(std::vector<lmminDataStruct> &optimizationDataStruct)
{
    float 
        img_scale = float( pow(2.0,double(pyr_levels_)) );
    
    for( int i = pyr_levels_; i >= 0; i--)
    {
        actual_scale_ = 1.0/img_scale;
        
        std::cout << "Starting opt a pyr layer..." << std::endl;
        optimize_all(i, optimizationDataStruct);
        std::cout << "Ended the opt of the layer" << std::endl;
        
        img_scale /= 2.0f;
    }
}

void NormalOptimizer::optimize_all(const int pyrLevel, std::vector<lmminDataStruct> &optimizationDataStruct)
{
    
    std::vector<lmminDataStruct>::iterator 
        dataIT = optimizationDataStruct.begin();
    
    int 
        n_par = 2 * optimizationDataStruct.size();
    
    double 
        par[n_par];
        
    m_dat_ = 0;
        
    int 
        index = 0;
    while (dataIT != optimizationDataStruct.end())
    {
        car2sph(*((*dataIT).normal), par[index], par[index + 1]);
        
//         std::cout << *((*dataIT).normal) << " - [" << par[index] << ", " << par[index + 1] << "]" << std::endl;
        
        dataIT->scale = actual_scale_;
        
        m_dat_ = m_dat_ + dataIT->m_dat;
        
        index = index + 2;
        dataIT++;
    }

    sct_->setImages(pyr_img_1_[pyrLevel],pyr_img_2_[pyrLevel]);
    
    /* auxiliary parameters */
    lm_status_struct 
    status;
    
    lm_control_struct 
        control = lm_control_double;
    control.epsilon = epsilon_lmmin_;
    control.patience = 20;
    
    lm_princon_struct 
        princon = lm_princon_std;
    princon.flags = 0;
    
    lmmin( n_par, par, m_dat_, &optimizationDataStruct, evaluateNormal,
           lm_printout_std, &control, 0/*&princon*/, &status );
    
    dataIT = optimizationDataStruct.begin();
    index = 0;
    while (dataIT != optimizationDataStruct.end())
    {
        sph2car(par[index], par[index + 1], *((*dataIT).normal));
        
        index = index + 2;
        dataIT++;
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
