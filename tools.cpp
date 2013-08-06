/**
 * \file tools.cpp
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

#include "tools.h"

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

cv::Scalar random_color(cv::RNG &rng)
{
    int color = rng.next();
    return CV_RGB(color&255, (color>>8)&255, (color>>16)&255);
}

/** Draw the matches and return the colors used
 */
void drawMatches(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &window, const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, const std::vector<cv::DMatch> &matches, std::vector<cv::Scalar> &colors, const std::vector<bool> outliersMask)
{
    window = cv::Mat(cv::Size(img1.cols * 2, img1.rows), CV_8UC3, cv::Scalar(0));
    
    cv::Mat
    img1BGR, img2BGR;
    
    cv::cvtColor(img1, img1BGR, CV_GRAY2BGR);
    cv::cvtColor(img2, img2BGR, CV_GRAY2BGR);
    
    img1BGR.copyTo(window(cv::Rect(0,0,img1.cols,img1.rows)));
    img2BGR.copyTo(window(cv::Rect(img1.cols,0,img2.cols,img2.rows)));
    
    cv::RNG 
    rng(0xFFFFFFFF);
    
    for (int i = 0; i < matches.size(); i++)
    {
        if (outliersMask[i])
        {
            cv::Scalar color = random_color(rng);
            colors.push_back(color);
            
            int
            idx1 = matches.at(i).queryIdx,
            idx2 = matches.at(i).trainIdx;
            
            cv::Point2f
            pt1 = kpts1.at(idx1).pt,
            pt2 = kpts2.at(idx2).pt;
            
            pt2.x = pt2.x + img1.cols;
            
            cv::circle(window, pt1, 4, color);
            cv::circle(window, pt2, 4, color);
            
            cv::line(window, pt1, pt2, color);
        }
//         else // All vectors must be N size!!
//         {
//             cv::Scalar color(0,0,0);
//             colors.push_back(color);
//         }
    }
}


void viewPointCloud(const cv::Mat &triagulatedPoints, const std::vector< cv::Scalar > &colors)
{
    ///////////////////////////// 
    // Converto i punti in point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    for (int i = 0; i < triagulatedPoints.cols; i++)
    {
        // Outliers are already out
        //         if ( outliersMask[i] ) // Do not print outliers
        //         {
        pcl::PointXYZRGB actual;
        actual.x = triagulatedPoints.at<double>(0, i);
        actual.y = triagulatedPoints.at<double>(1, i);
        actual.z = triagulatedPoints.at<double>(2, i);
        actual.r = colors.at(i)[2];
        actual.g = colors.at(i)[1];
        actual.b = colors.at(i)[0];
        
        cloud->points.push_back(actual);
        //         }
    }
    cloud->width = (int) cloud->points.size ();
    cloud->height = 1;
    
    ///////////////////////////// 
    // Visualizzo la point cloud
    boost::shared_ptr< pcl::visualization::PCLVisualizer >
    viewer( new pcl::visualization::PCLVisualizer("Triangulated points viewer") );
    
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Triangulated points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "Triangulated points");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    
    while (!viewer->wasStopped())
    {
        viewer->spin();
    }
}

void viewPointCloudNeighborhood(const cv::Mat &triagulatedPoints, std::vector< std::vector<cv::Vec3d> > &neighborhoodsVector, const std::vector< cv::Scalar > &colors)
{
    ///////////////////////////// 
    // Converto i punti in point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
        cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    for (int i = 0; i < triagulatedPoints.cols; i++)
    {
        pcl::PointXYZRGB actual;
        actual.x = triagulatedPoints.at<double>(0, i);
        actual.y = triagulatedPoints.at<double>(1, i);
        actual.z = triagulatedPoints.at<double>(2, i);
        actual.r = colors.at(i)[2];
        actual.g = colors.at(i)[1];
        actual.b = colors.at(i)[0];
        
        cloud->points.push_back(actual);
        
        for (std::size_t l = 0; l < neighborhoodsVector[i].size(); l++)
        {
            pcl::PointXYZRGB actualNeighbor;
            
            std::vector<cv::Vec3d>
                neighborhoods = neighborhoodsVector[i];
            
            cv::Vec3d
            point = neighborhoods[l];
            
            actualNeighbor.x = point[0];
            actualNeighbor.y = point[1];
            actualNeighbor.z = point[2];
            actualNeighbor.r = colors.at(i)[2];
            actualNeighbor.g = colors.at(i)[1];
            actualNeighbor.b = colors.at(i)[0];
            
            cloud->points.push_back(actualNeighbor);
        }
    }
    cloud->width = (int) cloud->points.size ();
    cloud->height = 1;
    
    ///////////////////////////// 
    // Visualizzo la point cloud
    boost::shared_ptr< pcl::visualization::PCLVisualizer >
    viewer( new pcl::visualization::PCLVisualizer("Triangulated points viewer") );
    
    viewer->setBackgroundColor(0,0,0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Triangulated points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "Triangulated points");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    
    while (!viewer->wasStopped())
    {
        viewer->spin();
    }
}