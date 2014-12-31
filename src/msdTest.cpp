
//This file is part of the MSD-Detector project (github.com/fedassa/msdDetector).
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
//along with the MSD-Detector project.If not, see <http://www.gnu.org/licenses/>.
// 
// AUTHOR: Federico Tombari (fedassa@gmail.com)
// University of Bologna, Open Perception

#include "opencv/cv.hpp"
#include "msd.h"

#include <time.h>
#include <iostream>

void main()
{
	//open a test image
	std::string filename = "../../../data/flinstones.png";
	cv::Mat img = cv::imread(filename, 0);
	
	MsdDetector msd;

	//Mode 1: calling MSD detector with default parameters
	std::vector<cv::KeyPoint> keypoints = msd.detect(img);
	
	//Mode 2: calling MSD detector with custom parameters
	//uncomment the following code and modify the default values where needed
	/*setPatchRadius(3);
	setSearchAreaRadius(5);

	setNMSRadius(5);
	setNMSScaleRadius(0);

	setThSaliency(250.0f);
	setKNN(4);

	setScaleFactor(1.25f);
	setNScales(-1);

	setComputeOrientation(false);

	std::vector<cv::KeyPoint> keypoints = msd.detect(img);
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