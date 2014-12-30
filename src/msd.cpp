

#include "msd.h"
#include <assert.h>
#define _USE_MATH_DEFINES
#include <math.h>

namespace msdDetector
{

	void convertMat2Array(cv::Mat &in, unsigned char * &out)
	{

		for (int j = 0; j<in.rows; j++)
		{
			for (int i = 0; i<in.cols; i++)
			{
				out[j*in.cols + i] = in.at<unsigned char>(j, i);
			}
		}

	}

	float computeAvgDistance(std::vector<int> &minVals, int den)
	{
		float avg_dist = 0.0f;
		for (int i = 0; i<(int)minVals.size(); i++)
			avg_dist += minVals[i];

		avg_dist = avg_dist / den;
		return avg_dist;
	}

	void contextualSelfDissimilarity_runningAvg(unsigned char *img, int w, int h, int xmin, int xmax, int r_s, int r_b, int k, float* saliency)
	{

		int side_s = 2 * r_s + 1;
		int side_b = 2 * r_b + 1;
		int border = r_s + r_b;
		int temp;
		int den = side_s * side_s * k;

		std::vector<int> minVals(k);
		int *acc = new int[side_b * side_b];
		int **vCol = new int *[w];
		for (int i = 0; i<w; i++)
			vCol[i] = new int[side_b * side_b];

		//first position
		int x = xmin;
		int y = border;

		int ctrInd = 0;
		for (int kk = 0; kk<k; kk++)
			minVals[kk] = std::numeric_limits<int>::max();

		for (int j = y - r_b; j <= y + r_b; j++)
		{
			for (int i = x - r_b; i <= x + r_b; i++)
			{
				if (j == y && i == x)
					continue;

				acc[ctrInd] = 0;
				for (int u = -r_s; u <= r_s; u++)
				{
					vCol[x + u][ctrInd] = 0;
					for (int v = -r_s; v <= r_s; v++)
					{
						temp = img[(j + v) * w + i + u] - img[(y + v) * w + x + u];
						vCol[x + u][ctrInd] += (temp*temp);
					}
					acc[ctrInd] += vCol[x + u][ctrInd];
				}

				if (acc[ctrInd]  < minVals[k - 1])
				{
					minVals[k - 1] = acc[ctrInd];

					for (int kk = k - 2; kk >= 0; kk--)
					{
						if (minVals[kk] > minVals[kk + 1])
						{
							std::swap(minVals[kk], minVals[kk + 1]);
						}
						else
							break;
					}
				}

				ctrInd++;
			}
		}
		saliency[y*w + x] = computeAvgDistance(minVals, den);

		for (x = xmin + 1; x<xmax; x++)
		{
			ctrInd = 0;
			for (int kk = 0; kk<k; kk++)
				minVals[kk] = std::numeric_limits<int>::max();

			for (int j = y - r_b; j <= y + r_b; j++)
			{
				for (int i = x - r_b; i <= x + r_b; i++)
				{
					if (j == y && i == x)
						continue;

					vCol[x + r_s][ctrInd] = 0;
					for (int v = -r_s; v <= r_s; v++)
					{
						temp = img[(j + v) * w + i + r_s] - img[(y + v) * w + x + r_s];
						vCol[x + r_s][ctrInd] += (temp*temp);
					}

					acc[ctrInd] = acc[ctrInd] + vCol[x + r_s][ctrInd] - vCol[x - r_s - 1][ctrInd];

					if (acc[ctrInd] < minVals[k - 1])
					{
						minVals[k - 1] = acc[ctrInd];
						for (int kk = k - 2; kk >= 0; kk--)
						{
							if (minVals[kk] > minVals[kk + 1])
							{
								std::swap(minVals[kk], minVals[kk + 1]);
							}
							else
								break;
						}
					}

					ctrInd++;
				}
			}
			saliency[y*w + x] = computeAvgDistance(minVals, den);
		}

		//all remaining rows...
		for (int y = border + 1; y< h - border; y++)
		{
			//first position of each row
			ctrInd = 0;
			for (int kk = 0; kk<k; kk++)
				minVals[kk] = std::numeric_limits<int>::max();
			x = xmin;

			for (int j = y - r_b; j <= y + r_b; j++)
			{
				for (int i = x - r_b; i <= x + r_b; i++)
				{
					if (j == y && i == x)
						continue;

					acc[ctrInd] = 0;
					for (int u = -r_s; u <= r_s; u++)
					{
						temp = img[(j + r_s) * w + i + u] - img[(y + r_s) * w + x + u];
						vCol[x + u][ctrInd] += (temp*temp);

						temp = img[(j - r_s - 1) * w + i + u] - img[(y - r_s - 1) * w + x + u];
						vCol[x + u][ctrInd] -= (temp*temp);

						acc[ctrInd] += vCol[x + u][ctrInd];
					}

					if (acc[ctrInd]  < minVals[k - 1])
					{
						minVals[k - 1] = acc[ctrInd];

						for (int kk = k - 2; kk >= 0; kk--)
						{
							if (minVals[kk] > minVals[kk + 1])
							{
								std::swap(minVals[kk], minVals[kk + 1]);
							}
							else
								break;
						}
					}

					ctrInd++;
				}
			}
			saliency[y*w + x] = computeAvgDistance(minVals, den);

			//all remaining positions
			for (x = xmin + 1; x<xmax; x++)
			{
				ctrInd = 0;
				for (int kk = 0; kk<k; kk++)
					minVals[kk] = std::numeric_limits<int>::max();

				for (int j = y - r_b; j <= y + r_b; j++)
				{
					for (int i = x - r_b; i <= x + r_b; i++)
					{
						if (j == y && i == x)
							continue;

						temp = img[(j + r_s)*w + i + r_s] - img[(y + r_s) * w + x + r_s];
						vCol[x + r_s][ctrInd] += (temp*temp);

						temp = img[(j - r_s - 1)*w + i + r_s] - img[(y - r_s - 1) * w + x + r_s];
						vCol[x + r_s][ctrInd] -= (temp*temp);

						acc[ctrInd] = acc[ctrInd] + vCol[x + r_s][ctrInd] - vCol[x - r_s - 1][ctrInd];

						if (acc[ctrInd] < minVals[k - 1])
						{
							minVals[k - 1] = acc[ctrInd];

							for (int kk = k - 2; kk >= 0; kk--)
							{
								if (minVals[kk] > minVals[kk + 1])
								{
									std::swap(minVals[kk], minVals[kk + 1]);
								}
								else
									break;
							}
						}
						ctrInd++;
					}
				}
				saliency[y*w + x] = computeAvgDistance(minVals, den);
			}
		}

		for (int i = 0; i<w; i++)
			delete[] vCol[i];
		delete[] vCol;
		delete[] acc;
	}

	float computeOrientation(cv::Mat &img, int x, int y, int patch_radius, int search_area_radius, std::vector<cv::Point2f> circle)
	{

		int temp;
		int w = img.cols;
		int h = img.rows;

		int side = search_area_radius * 2 + 1;
		int nBins = 36;
		float step = float((2 * M_PI) / nBins);
		std::vector<float> hist(nBins, 0);
		std::vector<int> dists(circle.size(), 0);

		int minDist = std::numeric_limits<int>::max();
		int maxDist = -1;

		for (int k = 0; k<(int)circle.size(); k++)
		{

			int j = y + static_cast <int> (circle[k].y);
			int i = x + static_cast <int> (circle[k].x);

			for (int v = -patch_radius; v <= patch_radius; v++)
			{
				for (int u = -patch_radius; u <= patch_radius; u++)
				{
					temp = img.at<unsigned char>(j + v, i + u) - img.at<unsigned char>(y + v, x + u);
					dists[k] += temp*temp;
				}
			}

			if (dists[k] > maxDist)
				maxDist = dists[k];
			if (dists[k] < minDist)
				minDist = dists[k];
		}

		float deltaAngle = 0.0f;
		for (int k = 0; k<(int)circle.size(); k++)
		{
			float angle = deltaAngle;
			float weight = (1.0f*maxDist - dists[k]) / (maxDist - minDist);

			float binF;
			if (angle >= 2 * M_PI)
				binF = 0.0f;
			else
				binF = angle / step;
			int bin = static_cast <int> (std::floor(binF));

			assert(bin >= 0 && bin < nBins);
			float binDist = abs(binF - bin - 0.5f);

			float weightA = weight * (1.0f - binDist);
			float weightB = weight * binDist;
			hist[bin] += weightA;

			if (2 * (binF - bin) < step)
				hist[(bin + nBins - 1) % nBins] += weightB;
			else
				hist[(bin + 1) % nBins] += weightB;

			deltaAngle += step;
		}

		int bestBin = -1;
		float maxBin = -1;
		for (int i = 0; i<nBins; i++)
		{
			if (hist[i] > maxBin)
			{
				maxBin = hist[i];
				bestBin = i;
			}
		}

		//parabolic interpolation
		int l = (bestBin == 0) ? nBins - 1 : bestBin - 1;
		int r = (bestBin + 1) % nBins;
		float bestAngle2 = bestBin + 0.5f * ((hist[l]) - (hist[r])) / ((hist[l]) - 2.0f*(hist[bestBin]) + (hist[r]));
		bestAngle2 = (bestAngle2 < 0) ? nBins + bestAngle2 : (bestAngle2 >= nBins) ? bestAngle2 - nBins : bestAngle2;
		bestAngle2 *= step;
		
		return bestAngle2;
	}

	std::vector<cv::KeyPoint> msd_detector(cv::Mat &img, int patch_radius, int search_area_radius, int nms_radius, int nms_scale_radius, float th_saliency, int kNN, float scale_factor, int n_scales, bool compute_orientation)
	{

		int w = img.cols;
		int h = img.rows;

		int border = search_area_radius + patch_radius;

		//aumatic computation of the number of scales
		if (n_scales == -1)
			n_scales = cvFloor(std::log(cv::min(img.cols, img.rows) / ((patch_radius + search_area_radius)*2.0 + 1)) / std::log(scale_factor));

		cv::Mat imgG;
		if (img.channels() == 1)
			imgG = img;
		else
			cv::cvtColor(img, imgG, CV_BGR2GRAY);
		
		std::vector<unsigned char *> scaleSpaceArray;
		std::vector<int> sswidths, ssheights;

		ImagePyramid scaleSpacer(imgG, n_scales, scale_factor);
		std::vector<cv::Mat> scaleSpaceGray = scaleSpacer.GetImPyrReadOnly();
		
		for (int r = 0; r < n_scales; r++)
		{
			unsigned char *imgT = new unsigned char[scaleSpaceGray[r].cols * scaleSpaceGray[r].rows];
			convertMat2Array(scaleSpaceGray[r], imgT);
			scaleSpaceArray.push_back(imgT);
			sswidths.push_back(scaleSpaceGray[r].cols);
			ssheights.push_back(scaleSpaceGray[r].rows);
		}

		std::vector<cv::KeyPoint> keypoints;
		std::vector<float *> saliency;
		saliency.resize(n_scales);

		for (int r = 0; r < n_scales; r++)
		{
			saliency[r] = new float[sswidths[r] * ssheights[r]];
		}

		for (int r = 0; r<n_scales; r++)
		{
#ifdef BOOST_MULTICORE
			unsigned nThreads = boost::thread::hardware_concurrency();
			unsigned stepThread = (sswidths[r] - 2 * border) / nThreads;

			std::vector<boost::thread*> threads;
			for (unsigned i = 0; i < nThreads - 1; i++)
			{
				threads.push_back(new boost::thread(contextualSelfDissimilarity_runningAvg, scaleSpaceArray[r], sswidths[r], ssheights[r], border + i*stepThread, border + (i + 1)*stepThread, patch_radius, search_area_radius, kNN, saliency[r]));
			}
			threads.push_back(new boost::thread(contextualSelfDissimilarity_runningAvg, scaleSpaceArray[r], sswidths[r], ssheights[r], border + (nThreads - 1)*stepThread, sswidths[r] - border, patch_radius, search_area_radius, kNN, saliency[r]));

			for (unsigned i = 0; i < threads.size(); i++)
			{
				threads[i]->join();
				delete threads[i];
			}
#else
			contextualSelfDissimilarity_runningAvg(scaleSpaceArray[r], sswidths[r], ssheights[r], border, sswidths[r] - border, patch_radius, search_area_radius, kNN, saliency[r]);
#endif
		}

		// ********** NMS STAGE **********
		cv::KeyPoint kp_temp;
		int side = search_area_radius * 2 + 1;

		// ********** COMPUTE LUT FOR CANONICAL ORIENTATION **********
		std::vector<cv::Point2f> orientPoints;
		if (compute_orientation)
		{
			int nBins = 36;
			float step = float((2 * M_PI) / nBins);
			float deltaAngle = 0.0f;

			for (int i = 0; i<nBins; i++)
			{
				cv::Point2f pt;
				pt.x = search_area_radius * cos(deltaAngle);
				pt.y = search_area_radius * sin(deltaAngle);

				orientPoints.push_back(pt);

				deltaAngle += step;
			}
		}

		for (int r = 0; r<n_scales; r++)
		{
			for (int j = border; j< ssheights[r] - border; j++)
			{
				for (int i = border; i< sswidths[r] - border; i++)
				{
					if (saliency[r][j * sswidths[r] + i] > th_saliency)
					{
						bool is_max = true;

						for (int k = cv::max(0, r - nms_scale_radius); k <= cv::min(n_scales - 1, r + nms_scale_radius); k++)
						{
							if (k != r)
							{
								int j_sc = cvRound(j * std::pow(scale_factor, r - k));
								int i_sc = cvRound(i * std::pow(scale_factor, r - k));

								if (saliency[r][j*sswidths[r] + i] < saliency[k][j_sc*sswidths[r] + i_sc])
								{
									is_max = false;
									break;
								}
							}
						}

						for (int v = cv::max(border, j - nms_radius); v <= cv::min(ssheights[r] - border - 1, j + nms_radius); v++)
						{
							for (int u = cv::max(border, i - nms_radius); u <= cv::min(sswidths[r] - border - 1, i + nms_radius); u++)
							{
								if (saliency[r][j*sswidths[r] + i] < saliency[r][v*sswidths[r] + u])
								{
									is_max = false;
									break;
								}
							}

							if (!is_max)
								break;
						}

						if (is_max)
						{
							kp_temp.pt.x = i * std::pow(scale_factor, r);
							kp_temp.pt.y = j * std::pow(scale_factor, r);
							kp_temp.response = saliency[r][j*sswidths[r] + i];
							kp_temp.size = (patch_radius*2.0f + 1) * std::pow(scale_factor, r);

							if (compute_orientation)
								kp_temp.angle = computeOrientation(scaleSpaceGray[r], i, j, patch_radius, search_area_radius, orientPoints);

							keypoints.push_back(kp_temp);
						}
					}
				}
			}
		}

		for (int r = 0; r<n_scales; r++)
		{
			delete[] scaleSpaceArray[r];
			delete[] saliency[r];
		}
		return keypoints;
	}

}