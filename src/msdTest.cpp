
//This file is part of the MSD-Detector project.
//
//The MSD-Detector is free software : you can redistribute it and / or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//The MSD-Detector is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with Foobar.If not, see <http://www.gnu.org/licenses/>.
// 
// AUTHOR: Federico Tombari (fedassa@gmail.com)
// University of Bologna, Open Perception

#include "opencv/cv.hpp"
#include "msd.h"

#include <iostream>

void main()
{
	//open a test image
	std::string filename = "../../../data/flinstones.png";
	cv::Mat img = cv::imread(filename, 0);
	
	//Mode 1: calling MSD detector with default parameters
	std::vector<cv::KeyPoint> keypoints = msdDetector::msd_detector(img);

	//Mode 2: calling MSD detector with custom parameters - uncomment the following code
	/*int patch_radius = 3;
	int search_area_radius = 5;

	int nms_radius = 5;
	int nms_scale_radius = 0;

	float th_saliency = 250.0f;
	int kNN = 4;

	float scale_factor = 1.25;
	int n_scales = -1;
	bool compute_orientation = false;

	std::vector<cv::KeyPoint> keypoints = msdDetector::msd_detector(img, patch_radius, search_area_radius, nms_radius, nms_scale_radius, th_saliency, kNN, scale_factor, n_scales, compute_orientation);
	*/

	std::cout << "Found " << keypoints.size() << " keypoints" << std::endl;

	//visualize detected keypoints
	cv::Mat kpsImg; 
	cv::cvtColor(img, kpsImg, CV_GRAY2BGR);

	for (unsigned int i = 0; i<keypoints.size(); i++)
	{
		cv::circle(kpsImg, keypoints[i].pt, cvRound((keypoints[i].size-1)/2), cv::Scalar(0, 255, 0), 1);
	}

	cv::imshow("MSD Keypoints", kpsImg);
	cv::waitKey(0);

}