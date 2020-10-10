#ifndef RANDOM_WARPED_IMAGE_H
#define RANDOM_WARPED_IMAGE_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <TRandomDouble.h>

///Возвращает случайную матрицу преобразования
inline cv::Mat GetRandomWarpMatrix(
	const cv::Size &src_size,
	const float &width_variation_factor,
	const float &height_variation_factor
){
	const float delta = (float)0.03;
	cv::Mat warp_matrix;//(3,3,CV_32FC1);
	cv::Point2f src_quad[4], dst_quad[4];
		
	//Задаём точки
	src_quad[0].x = 0;
	src_quad[0].y = 0;
	src_quad[1].x = (float)src_size.width - 1;
	src_quad[1].y = 0;
	src_quad[2].x = 0;
	src_quad[2].y = (float)src_size.height - 1;
	src_quad[3].x = (float)src_size.width - 1;
	src_quad[3].y = (float)src_size.height - 1;

	dst_quad[0].x = src_size.width * (float)RandomDouble() * width_variation_factor * ((float)1 - delta);
	dst_quad[0].y = src_size.height * (float)RandomDouble() * height_variation_factor * ((float)1 - delta);
	dst_quad[1].x = src_size.width * ((float)1. - (float)RandomDouble() * width_variation_factor) * ((float)1 + delta);
	dst_quad[1].y = src_size.height * (float)RandomDouble() * height_variation_factor * ((float)1 - delta);
	dst_quad[2].x = src_size.width * (float)RandomDouble() * width_variation_factor * ((float)1 - delta);
	dst_quad[2].y = src_size.height * ((float)1. - (float)RandomDouble() * height_variation_factor) * ((float)1 + delta); 
	dst_quad[3].x = src_size.width * ((float)1. - (float)RandomDouble() * width_variation_factor) * ((float)1 + delta);
	dst_quad[3].y = src_size.height * ((float)1.- (float)RandomDouble() * height_variation_factor) * ((float)1 + delta);

	//Получаем матрицу преобразования
	warp_matrix = cv::getPerspectiveTransform(src_quad, dst_quad);
	return warp_matrix;
}

//Point spread function
inline void PSF(cv::Mat &outputImg, cv::Size filterSize, int len, double theta)
{
	cv::Mat h(filterSize, CV_32F, cv::Scalar(0));
	cv::Point point(filterSize.width / 2, filterSize.height / 2);
	cv::ellipse(h, point, cv::Size(0, cvRound(float(len) / 2.0)), 90.0 - theta, 0, 360, cv::Scalar(255), cv::FILLED);
	cv::Scalar summa = sum(h);
	outputImg = h / summa[0];
}

inline void RandomMotionBlur(const cv::Mat &src, cv::Mat &dst, int max_len)
{
	int len = RandomInt() % max_len;
	double theta = RandomInt() % 360;
	cv::Mat psf;
	PSF(psf, cv::Size(max_len, max_len), len, theta);
	cv::filter2D(src, dst, src.type(), psf);
}

inline void PixelNoise(cv::Mat &img, const float &max_brightness_variation_factor)
{
	const float brightness_variation_factor = (float)RandomDouble() * max_brightness_variation_factor;
	//Вариация яркости - случайный шум
	for (int i = 0; i < img.size().width; i++)
		for (int j = 0; j < img.size().height; j++)
		{
			if (img.type() == CV_8UC1)
			{
				int t = (int)img.at<unsigned char>(j, i) + (int)(brightness_variation_factor * 256 * ((float)RandomDouble() - (float)0.5));
				if (t < 0) t = 0;
				if (t > 255) t = 255;
				img.at<unsigned char>(j, i) = t;
			}
			if (img.type() == CV_8UC3)
			{
				int t;
				for (int k = 0; k < 3; k++)
				{
					t = (int)img.at<cv::Vec3b>(j, i)[k] + (int)(brightness_variation_factor * 256 * ((float)RandomDouble() - (float)0.5));
					if (t < 0) t = 0;
					if (t > 255) t = 255;
					img.at<cv::Vec3b>(j, i)[k] = t;
				}
			}
			if (img.type() != CV_8UC1 && img.type() != CV_8UC3) throw TException("PixelNoise error: incorrect mat type");
		}
}


inline void Terminator(const cv::Mat &src, cv::Mat &dst, const double &alpha_max)
{
	cv::Mat temp;
	unsigned long	side1 = RandomInt() % 4,
		side2 = RandomInt() % 4;

	if (src.empty())
		throw std::exception("Terminator error empty source!");
	if (src.cols == 0 || src.rows == 0)
		throw std::exception("Terminator error incorrect input!");

	temp = cv::Mat(src.size(), src.type());
	temp = 0;

	while (side1 == side2)
		side2 = RandomInt() % 4;

	std::vector<cv::Point>	corners,
		points;
	cv::Point p;

	corners.push_back(cv::Point(0, 0));
	corners.push_back(cv::Point(src.cols, 0));
	corners.push_back(cv::Point(src.cols, src.rows));
	corners.push_back(cv::Point(0, src.rows));

	p = corners[side1];
	if (side1 == 0 || side1 == 2)
		p.x = RandomInt() % src.cols;
	if (side1 == 1 || side1 == 3)
		p.y = RandomInt() % src.rows;
	points.push_back(p);

	if (side2 - side1 == 1 || side2 - side1 == -3)
		points.push_back(corners[(side1 + 1) % 4]);

	if (abs((int)(side2 - side1)) == 2)
	{
		points.push_back(corners[(side1 + 1) % 4]);
		points.push_back(corners[(side1 + 2) % 4]);
	}

	if (side2 - side1 == -1 || side2 - side1 == 3)
	{
		points.push_back(corners[(side1 + 1) % 4]);
		points.push_back(corners[(side1 + 2) % 4]);
		points.push_back(corners[(side1 + 3) % 4]);
	}

	p = corners[side2];
	if (side2 == 0 || side2 == 2)
		p.x = RandomInt() % src.cols;
	if (side2 == 1 || side2 == 3)
		p.y = RandomInt() % src.rows;
	points.push_back(p);

	cv::fillConvexPoly(temp, points, cv::Scalar(255, 255, 255));

	double	alpha = 1. - RandomDouble() * alpha_max,
		beta;
	beta = (1.0 - alpha);
	addWeighted(src, alpha, temp, beta, 0.0, dst);
}


inline void RandomWarp(
	const cv::Mat &src, 
	cv::Mat &dst, 
	const cv::Mat warp_matrix, 	
	const bool calc_background = false 
){
	//Вычисление среднего цвета по границе изображения
	cv::Scalar c = cvScalarAll(0);
	if (calc_background)
	{
		float c1 = 0, 
					c2 = 0, 
					c3 = 0,
					n = (float)2.*src.size().width + (float)2.*src.size().height;
		for (int i = 0; i < src.size().width; i++)
		{
			if (src.type() == CV_8UC3)
			{
				c1 += src.at<cv::Vec3b>(0, i)[0];
				c2 += src.at<cv::Vec3b>(0, i)[1];
				c3 += src.at<cv::Vec3b>(0, i)[2];

				c1 += src.at<cv::Vec3b>(src.size().height - 1, i)[0];
				c2 += src.at<cv::Vec3b>(src.size().height - 1, i)[1];
				c3 += src.at<cv::Vec3b>(src.size().height - 1, i)[2];
			};

			if (src.type() == CV_8UC1)
			{
				c1 += src.at<unsigned char>(0, i);

				c1 += src.at<unsigned char>(src.size().height - 1, i);
			};
		};

		for (int i = 0; i < src.size().height; i++)
		{
			if (src.type() == CV_8UC3)
			{
				c1 += src.at<cv::Vec3b>(i, 0)[0];
				c2 += src.at<cv::Vec3b>(i, 0)[1];
				c3 += src.at<cv::Vec3b>(i, 0)[2];

				c1 += src.at<cv::Vec3b>(i, src.size().width - 1)[0];
				c2 += src.at<cv::Vec3b>(i, src.size().width - 1)[1];
				c3 += src.at<cv::Vec3b>(i, src.size().width - 1)[2];
			};

			if (src.type() == CV_8UC1)
			{
				c1 += src.at<unsigned char>(i, 0);

				c1 += src.at<unsigned char>(i, src.size().width - 1);
			};
		};

		if (src.type() == CV_8UC3) c = cv::Scalar(c1/n, c2/n, c3/n, 0);
		if (src.type() == CV_8UC1) c = cv::Scalar(c1/n);
	};

	//Преобразование перспективы
	warpPerspective(src, dst, warp_matrix, src.size(), 1, 0, c);
}

///Возвращает случайно перспективно трансформированное изображение
inline cv::Mat GetRandomWarpedImage(
	const cv::Mat &src,
	const float &width_variation_factor,
	const float &height_variation_factor,
	const float &brightness_variation_factor,
	const bool flip_vertical,
	const bool flip_horizontal,
	const bool calc_background /*= false*/,
	const bool add_terminator,
	const double terminator_alpha_max,
	const int max_motion_blur_len
){
	//float bf = brightness_variation_factor * (RandomDouble() - 0.5);
	cv::Mat warp_matrix = GetRandomWarpMatrix(src.size(), width_variation_factor, height_variation_factor), //(3,3,CV_32FC1);	//Получаем матрицу преобразования
					result;

	int flip_code;
	if (flip_vertical)
		flip_code = 1;
	if (flip_horizontal)
		flip_code = 0;
	if (flip_vertical && flip_horizontal)
	{
		if (RandomDouble() > 0.5)
			flip_code = 0;
		else
			flip_code = 1;
	}

	if (RandomDouble() > 0.5)
	{
		cv::flip(src, result, flip_code);

		if (add_terminator)
			Terminator(result, result, terminator_alpha_max);
	} else {
		if (add_terminator)
			Terminator(src, result, terminator_alpha_max);
	}

	RandomWarp(result, result, warp_matrix, calc_background);
	RandomMotionBlur(result, result, max_motion_blur_len);
	PixelNoise(result, brightness_variation_factor);
	return result;
};

#endif