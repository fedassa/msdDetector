

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