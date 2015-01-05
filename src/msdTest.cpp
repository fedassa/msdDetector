
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

int main()
{
	//open a test image
#ifdef WIN32
	std::string filename = "../../../data/flinstones.png";
#else
	std::string filename = "../../data/flinstones.png";
#endif
	cv::Mat img = cv::imread(filename, 0);
	
	MsdDetector msd;
	
	//If you want to use custom MSD parameters rather than using the default ones, 
	//uncomment the following code and modify the default values where needed
	/*msd.setPatchRadius(3);
	msd.setSearchAreaRadius(5);

	msd.setNMSRadius(5);
	msd.setNMSScaleRadius(0);

	msd.setThSaliency(250.0f);
	msd.setKNN(4);

	msd.setScaleFactor(1.25f);
	msd.setNScales(-1);

	msd.setComputeOrientation(false);*/

	std::vector<cv::KeyPoint> keypoints = msd.detect(img);

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

	return 0;
}