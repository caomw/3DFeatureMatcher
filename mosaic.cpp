/**
 * \file mosaic.cpp
 * \Author: Michele Marostica
 *  
 * Copyright (c) 2012, Michele Marostica (michelemaro@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 * 1 - Redistributions of source code must retain the above copyright notice, 
 *     this list of conditions and the following disclaimer.
 * 2 - Redistributions in binary form must reproduce the above copyright 
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
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

#include "mosaic.h"

MOSAIC::MOSAIC(cv::FileStorage fs, 
               cv::Mat& imgA, cv::Mat& imgB, 
               const cv::Vec3d tA, const cv::Vec3d tB, 
               const cv::Vec3d rA, const cv::Vec3d rB)
{
    // 1 - Set the images
    imgA_ = imgA;
    imgB_ = imgB;
    
    // 2 - Compute the matches
    cv::Mat
        desc1, desc2;
    dm_ = new DescriptorsMatcher(fs, *imgA_, *imgB_);
    dm_->compareWithNNDR(fs["NNDR"]["epsilon"], matches_, kptsA_, kptsB_, desc1, desc2);
    
    // 3 - Set the transformation matrix between poses A and B
    sct_ = new SingleCameraTriangulator(fs);
    sct_->setg12(tA, tB, rA, rB, gAB_);
    
    // 4 - Compute the triangulation
    std::vector<bool>
        outliersMask;   // Mask to distinguish between inliers (1) and outliers (0) match points. 
                        // As example those with negative z are outliers.
    sct_->setKeypoints(kptsA_, kptsB_, matches_);
    sct_->triangulate(triangulated_points_, outliersMask); // 3D points are all inliers! The mask is for the matches
    
    // 5 - Setup the normal optimzer, compute the normals and the features frames
    no_ = new NormalOptimizer(fs, sct_);
    no_->setImages(*imgA_, *imgB_);
    no_->computeOptimizedNormals(triangulated_points_, normals_vector_, colors_);
    no_->computeFeaturesFrames(triangulated_points_, normals_vector_, features_frames_);
    
    // 6 - Setup the neighborhood generator
    ng_ = new NeighborhoodsGenerator(fs);
    ng_->getReferenceSquaredNeighborhood(reference_neighborhood_);
    
    // 7 - Obtain the patches
    sct_->setImages(*imgA_, *imgB_);
    sct_->projectReferencePointsToImageWithFrames(reference_neighborhood_, features_frames_, patches_vector_, image_points_vector_);
    
    ///...
}

void MOSAIC::computeImpl(const cv::Mat& image, std::vector< cv::KeyPoint, std::allocator< void > >& keypoints, cv::Mat& descriptors)
{
    /// Usare un metodo setImages per passare frame A e B e ignorare image e keypoints
    /// verificare cosa passare alla funzione compute affiché non scazzi tutto. DC
}

int MOSAIC::descriptorType()
{

}

int MOSAIC::descriptorSize()
{

}
