/**
 * \file normaloptimizer.h
 * \Author: Michele Marostica
 * \brief: This class has the purpose to find optimal 3D normals of the matches of features extracted
 *         from 2 images. The optimization is by minimizing the intensity reprojection error of the 
 *         neighborhood of a feature keypoint in image 1 reprojected in image 2.
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

#ifndef NORMALOPTIMIZER_H
#define NORMALOPTIMIZER_H

#include <opencv2/opencv.hpp>
#include <lmmin.h>
#include <tr1/memory>

#include "singlecameratriangulator.h"
#include "../pclvisualizerthread.h"
#include "../tools.h"



typedef struct lmminDataStruct_{
    
    SingleCameraTriangulator
        *sct;
    cv::Vec3d
        *point,
        *normal;
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
    bool
        isGood;
    
} lmminDataStruct;

class NormalOptimizer
{
public:
    NormalOptimizer(const cv::FileStorage settings, SingleCameraTriangulator *sct);

    void setImages(const cv::Mat &img1, const cv::Mat &img2);
    
    void computeOptimizedNormals(std::vector< cv::Vec3d >& points3D, std::vector< cv::Vec3d >& normalsVector);
    void computeOptimizedNormals(std::vector< cv::Vec3d >& points3D, std::vector< cv::Vec3d >& normalsVector, std::vector<cv::Scalar> &colors);
    
    void computeOptimizedNormalsAllInOne(std::vector< cv::Vec3d >& points3D, std::vector< cv::Vec3d >& normalsVector, std::vector<cv::Scalar> &colors);
    
    void computeFeaturesFrames(std::vector< cv::Vec3d >& points3D, std::vector< cv::Vec3d >& normalsVector, std::vector<cv::Matx44d> &featuresFrames);
    
    void startVisualizerThread();
    
    void stopVisualizerThread();
    
    cv::Vec3d getGravity();
    
// private methods
private:
    NormalOptimizer(); // Avoid default constructor
    
    void compute_pyramids(const cv::Mat &img1, const cv::Mat &img2);
    
    bool optimize(const int pyrLevel);
    
    bool optimize_pyramid();
    
    void optimize_all(const int pyrLevel, std::vector<lmminDataStruct> &optimizationDataStruct);
    
    void optimize_pyramid_all(std::vector<lmminDataStruct> &optimizationDataStruct);
    
// private data
private:
    
    int
        pyr_levels_;
    
    std::vector<cv::Mat>
        pyr_img_1_,
        pyr_img_2_;
        
    SingleCameraTriangulator
        *sct_;
        
    cv::Ptr< const cv::Vec3d >
        gravity_;
        
    /*LMMIN parameters*/
    double
        epsilon_lmmin_;
        
    int
        m_dat_;
        
    double
        actual_scale_;
        
    double
        fixed_angle_; //for wall features
    /*LMMIN parameters*/
    
    boost::thread 
        *workerThread_;
    pclVisualizerThread
        *visualizer_;
};

#endif // NORMALOPTIMIZER_H
