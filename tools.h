/**
 * \file tools.h
 * \Author: Michele Marostica
 * \brief: Some utilities functions
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

#ifndef TOOLS_H_
#define TOOLS_H_

#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/thread/thread.hpp>

/** Compose a 4x4 transform matrix from R and T
 */
void composeTransformation(const cv::Matx33d &R, const cv::Vec3d &T, cv::Matx44d &G);

/** Generate a random color (RGB)
 */
cv::Scalar random_color(cv::RNG &rng);

/** Draw the matches and return the colors used
 */
void drawMatches(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &window, const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, const std::vector<cv::DMatch> &matches, std::vector<cv::Scalar> &colors, const std::vector<bool> outliersMask);

/** Draw a point cloud
 */
void viewPointCloud(const cv::Mat &triagulatedPoints, const std::vector< cv::Scalar > &colors);

void viewPointCloudNeighborhood(const cv::Mat &triagulatedPoints, std::vector< std::vector<cv::Vec3d> > &neighborhoodsVector, const std::vector< cv::Scalar > &colors);
#endif // TOOLS_H_