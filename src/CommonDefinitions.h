#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

#include <dlib/opencv/cv_image.h>
#include <dlib/data_io.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

#ifdef USE_MXNET
#if defined(_MSC_VER)
#define DISABLE_WARNING_PUSH           __pragma(warning( push ))
#define DISABLE_WARNING_POP            __pragma(warning( pop )) 
#define DISABLE_WARNING(warningNumber) __pragma(warning( disable : warningNumber ))

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER    DISABLE_WARNING(4100)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION            DISABLE_WARNING(4505)
// other warnings you want to deactivate...

#elif defined(__GNUC__) || defined(__clang__)
#define DO_PRAGMA(X) _Pragma(#X)
#define DISABLE_WARNING_PUSH           DO_PRAGMA(GCC diagnostic push)
#define DISABLE_WARNING_POP            DO_PRAGMA(GCC diagnostic pop) 
#define DISABLE_WARNING(warningName)   DO_PRAGMA(GCC diagnostic ignored #warningName)

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER    DISABLE_WARNING(-Wunused-parameter)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION            DISABLE_WARNING(-Wunused-function)
// other warnings you want to deactivate... 

#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
// other warnings you want to deactivate... 

#endif


DISABLE_WARNING_PUSH

#if defined(_MSC_VER)
DISABLE_WARNING(4267)
DISABLE_WARNING(4305)
DISABLE_WARNING(4251)
DISABLE_WARNING(4244)
#endif

#include "mxnet-cpp/MxNetCpp.h"

DISABLE_WARNING_POP
#endif

#ifdef QT_VERSION
#include <string>
#include <QtCore/QDebug>
#include <QtGui/QImage>
#include <QtGui/QPixmap>

#include <WriteToLog.h>

///Преобразование cv::Mat в QImage
inline QImage cvMatToQImage(const cv::Mat &mat)
{
  switch (mat.type())
  {
    // 8-bit, 4 channel
    case CV_8UC4:
    {
      QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_ARGB32);
      return image;
    }

    // 8-bit, 3 channel
    case CV_8UC3:
    {
      QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_RGB888);
      return image.rgbSwapped();
    }

    // 8-bit, 1 channel
    case CV_8UC1:
    {
      #if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
      QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8);
      #else
      static QVector<QRgb>  sColorTable;
      // only create our color table the first time
      if (sColorTable.isEmpty())
      {
        sColorTable.resize(256);
        for (int i = 0; i < 256; ++i) sColorTable[i] = qRgb(i, i, i);
      }
      QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_Indexed8);
      image.setColorTable(sColorTable);
      #endif
      return image;
    }

    default:
      throw TException("cvMatToQImage() error: cv::Mat image type not supported: " /*+ std::to_string(mat.type())*/);
    break;
  }
  return QImage();
}

///Преобразование cv::Mat в QPixmap
inline QPixmap cvMatToQPixmap(const cv::Mat &mat)
{
  return QPixmap::fromImage(cvMatToQImage(mat));
}

#endif  //#ifdef QT_VERSION

inline void rotateImage(const cv::Mat &in, cv::Mat &out, const double &angle)
{
  // get rotation matrix for rotating the image around its center
  cv::Point2f center(in.cols/2.0, in.rows/2.0);
  cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
  // determine bounding rectangle
  cv::Rect bbox = cv::RotatedRect(center, in.size(), angle).boundingRect();
  // adjust transformation matrix
  rot.at<double>(0,2) += bbox.width/2.0 - center.x;
  rot.at<double>(1,2) += bbox.height/2.0 - center.y;

  cv::warpAffine(in, out, rot, bbox.size());
}

///
static void GetCorners(
	cv::Mat H,
	int width,
	int height,
	std::vector<cv::Point2f> &sceneCorners
)
{
	assert(!H.empty());
	std::vector<cv::Point2f> objCorners = {
		cv::Point(0, 0),
		cv::Point(width, 0),
		cv::Point(width, height),
		cv::Point(0, height)
	};

	cv::perspectiveTransform(objCorners, sceneCorners, H);
}

///Преобразование в градации серого
static void ToGrayscale(const cv::Mat &image, cv::Mat &gray)
{
	if (image.type() != CV_8UC1)
	{
		cv::cvtColor(image, gray, CV_BGR2GRAY);
		gray.convertTo(gray, CV_8UC1);
	} else {
		image.copyTo(gray);
	}
}

///Преобразование матрицы чисел с плавающей запятой из формата opencv в формат dlib
//В обратную сторону cv::Mat cv_mat = dlib::toMat(dlib_matrix);
static void CVMatToDlibMatrixFC1(const cv::Mat &mat, dlib::matrix<float> &dlib_matrix)
{
	cv::Mat temp(mat.rows, mat.cols, CV_32FC1);
	cv::normalize(mat, temp, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
	dlib::assign_image(dlib_matrix, dlib::cv_image<float>(temp));
}

///Преобразование матрицы целых чисел из формата opencv в формат dlib
//В обратную сторону cv::Mat cv_mat = dlib::toMat(dlib_matrix);
static void CVMatToDlibMatrix8U(const cv::Mat &mat, dlib::matrix<unsigned char> &dlib_matrix)
{
	cv::Mat temp(mat.rows, mat.cols, CV_8U);
	cv::normalize(mat, temp, 0, 255, cv::NORM_MINMAX, CV_8U);
	dlib::assign_image(dlib_matrix, dlib::cv_image<unsigned char>(temp));
}

#ifdef USE_MXNET

//convert cv::Mat to mxnet::cpp::NDArray format
//!!!Передевать NDArray параметром, если его размеры удовлетворяют, то не производить выделения памяти
//!!!Если порядок совпадает, то memcpy
static mxnet::cpp::NDArray CVMatToMXNDArray(const cv::Mat &mat, const mxnet::cpp::Context &context)
{
	//if (nd_array.GetShape() )
	mxnet::cpp::NDArray nd_array(mxnet::cpp::Shape(1, mat.channels(), mat.cols, mat.rows), context, false);
	std::vector<mx_float> data;
	data.reserve(1 * mat.channels() * mat.rows * mat.cols);

	for (int c = 0; c < mat.channels(); c++)
		for (int i = 0; i < mat.rows; i++) 
			for (int j = 0; j < mat.cols; j++) 
			{
				data.emplace_back(static_cast<mx_float>(mat.data[(i * mat.cols + j) * mat.channels() + c]));
			}

	nd_array.SyncCopyFromCPU(data.data(), 1 * mat.channels() * mat.rows * mat.cols);
	mxnet::cpp::NDArray::WaitAll();
	return nd_array;
}


///nd_array should already be not in GPU 
///result image type can be float 1 channel CV_32FC1 or float 3 channel CV_32FC3
static cv::Mat MXNDArrayToCVMat(const mxnet::cpp::NDArray nd_array, mxnet::cpp::Shape shape = mxnet::cpp::Shape()/*, const mxnet::cpp::Context &context*/)
{
	//data_shape = mxnet::cpp::Shape(Properties.BatchSize, 1, Properties.DatasetImageWidth, Properties.DatasetImageHeight);
	//label_shape = mxnet::cpp::Shape(Properties.BatchSize);
	
	if (shape == mxnet::cpp::Shape())
		shape = nd_array.GetShape();

	const mx_float * data = nd_array.GetData();

	int img_type;
	if (shape[1] == 1)
		img_type = CV_32FC1;
	else if (shape[1] == 3)
		img_type = CV_32FC3;
	else
		std::runtime_error("MXNDArrayToCVMat error: the number of channels of the image is expected to be 1 or 3");

	cv::Mat result(shape[3], shape[2], img_type);

	int k = 0;
	for (int c = 0; c < result.channels(); c++)
		for (int i = 0; i < result.rows; i++)
			for (int j = 0; j < result.cols; j++)
			{
				*((float*)&(result.data[((i * result.cols + j) * result.channels() + c)*sizeof(float)])) = data[k];
				//result.data[k] = data[(i * result.rows + j) * result.channels() + c];
				k++;
			}
	return result;
}

//TEST
//mxnet::cpp::Context context_gpu(mxnet::cpp::DeviceType::kGPU, 0),
//context_cpu(mxnet::cpp::DeviceType::kCPU, 0);
//MXSetNumOMPThreads(std::thread::hardware_concurrency());
//cv::Mat m = cv::imread("D:/MMedia/Pictures/_icvfAkQX3k.jpg");
//std::cout << m.cols << " " << m.rows << std::endl;
//mxnet::cpp::NDArray nda1 = CVMatToMXNDArray(m, context_gpu);
//auto shape1 = nda1.GetShape();
//for (size_t i = 0; i < shape1.size(); i++)
//	std::cout << shape1[i] << " ";
//std::cout << std::endl;
//std::cout << nda1.GetContext().GetDeviceType() << std::endl;
//mxnet::cpp::NDArray nda2 = nda1.Copy(context_cpu);
//mxnet::cpp::NDArray::WaitAll();
//auto shape2 = nda2.GetShape();
//for (size_t i = 0; i < shape2.size(); i++)
//	std::cout << shape2[i] << " ";
//std::cout << std::endl;
//std::cout << nda2.GetContext().GetDeviceType() << std::endl;
//cv::Mat mat = MXNDArrayToCVMat(nda2);
//std::cout << mat.cols << " " << mat.rows << std::endl;
//cv::Mat temp(mat.rows, mat.cols, CV_8U);
//cv::normalize(mat, temp, 0, 255, cv::NORM_MINMAX, CV_8U);

#endif //USE_MXNET

#endif