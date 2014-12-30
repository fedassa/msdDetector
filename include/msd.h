
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

#ifndef LSD_DETECTOR_H_
#define LSD_DETECTOR_H_

#include <vector>
#include "opencv/cv.hpp"

#include "msdImgPyramid.h"


//#define BOOST_MULTICORE

#ifdef BOOST_MULTICORE
#include "boost\thread.hpp"
#endif

namespace msdDetector
{
	std::vector<cv::KeyPoint> msd_detector(cv::Mat &img, int patch_radius=3, int search_area_radius=5, int nms_radius=5, int nms_scale_radius=0, float th_saliency=250.0f, int kNN=4, float scale_factor=1.25, int n_scales = -1, bool compute_orientation = false);	
}

#endif