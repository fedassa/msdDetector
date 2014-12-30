

#ifndef IMG_PYRAMID_H_
#define IMG_PYRAMID_H_

#include "opencv/cv.hpp"

class ImagePyramid
{
public:

	ImagePyramid(const cv::Mat &im, const int nLevels, const float scaleFactor = 1.6f);
	~ImagePyramid();

	const std::vector<cv::Mat> GetImPyrReadOnly() const { return m_imPyr; };

private:

	std::vector<cv::Mat> m_imPyr;
	int m_nLevels;
	float m_scaleFactor;
};

#endif