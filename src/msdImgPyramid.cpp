
#include "msdImgPyramid.h"

ImagePyramid::ImagePyramid(const cv::Mat & im, const int nLevels, const float scaleFactor)
{
	m_nLevels = nLevels;
	m_scaleFactor = scaleFactor;
	m_imPyr.clear();
	m_imPyr.resize(nLevels);

	m_imPyr[0] = im.clone();

	if(m_nLevels > 1)
	{	
		for (int lvl = 1; lvl < m_nLevels; lvl++)
		{
			float scale = 1 / std::pow(scaleFactor, (float)lvl);
			m_imPyr[lvl] = cv::Mat(cv::Size(cvRound(im.cols * scale), cvRound(im.rows * scale)), im.type());
			cv::resize(im, m_imPyr[lvl], cv::Size(m_imPyr[lvl].cols, m_imPyr[lvl].rows), 0.0, 0.0, CV_INTER_AREA);
		}
	}
}

ImagePyramid::~ImagePyramid()
{
}







