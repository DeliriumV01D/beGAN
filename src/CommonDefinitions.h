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
  cv::Point2f center(in.cols/2.0f, in.rows/2.0f);
  cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
  // determine bounding rectangle
  cv::Rect bbox = cv::RotatedRect(center, in.size(), (float)angle).boundingRect();
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

inline cv::Mat Colorize(const cv::Mat& img)
{
	const std::array<cv::Vec3b, 4> colors =
	{
		cv::Vec3b(107, 107, 165),			//ground #a56b6b
		cv::Vec3b(245, 61, 184),			//structure #b83df5
		cv::Vec3b(64, 117, 35),				//forest #237540
		cv::Vec3b(231, 180, 49)				//water #31b4e7
	};

	cv::Mat result = cv::Mat(img.rows, img.cols, CV_8UC3);
	for (int i = 0; i < img.size().area(); i++)
	{
		int c;
		if (img.depth() == CV_8U)
			c = static_cast<int>(img.at<uchar>(i));
		if (img.depth() == CV_32F)
			c = static_cast<int>(round(img.at<float>(i)));
		if (c < 0 || c >= colors.size())
			result.at<cv::Vec3b>(i) = cv::Vec3b(0, 0, 0);
		else
			result.at<cv::Vec3b>(i) = colors[c];
	}
	return result;
}

inline cv::Mat Unfold(const cv::Mat& mat, const int channels)
{
	cv::Mat unfolded(mat.size(), CV_8UC(channels));
	//bk_mask.copyTo(mask, (man_mask >= 0 && man_mask < 1));
	for (int im = 0; im < mat.rows; im++)
		for (int jm = 0; jm < mat.cols; jm++)
		{
			for (int cm = 0; cm < channels; cm++)
				unfolded.data[(im * mat.cols + jm) * unfolded.channels() + cm] = 0;

			uchar uc = mat.at<uchar>(im, jm);
			if (uc >= 0 && uc < channels)
				unfolded.data[(im * mat.cols + jm) * unfolded.channels() + uc] = 1;

			//*((float*)&unfolded_label.data[((im * label.cols + jm) * unfolded_label.channels() + uc)*sizeof(float)]) = 1.f;
		}
	return unfolded;
}

///input image must be a multichannel float mat CV_32FC(shape[1])
inline cv::Mat Fold(const cv::Mat& mat)
{
	cv::Mat result(mat.size(), CV_32FC1);

	for (int im = 0; im < mat.rows; im++)
		for (int jm = 0; jm < mat.cols; jm++)
		{
			//auto v = mat.at<cv::Vec<float, mat.channels>>(im, jm);
			result.at<float>(im, jm) = 0.f;
			for (int cm = 0; cm < mat.channels(); cm++)
				if (*((float*)&mat.data[((im * mat.cols + jm) * mat.channels() + cm) * mat.elemSize1()]) > 0.5)
				{
					result.at<float>(im, jm) = static_cast<float>(cm);
				}
		}
	return result;
}

#ifdef USE_MXNET

inline void CVMatToMxFloatArr(
	const cv::Mat &mat,											//отсюда
	mx_float * data,												//сюда без проверки границ
	cv::Rect roi = cv::Rect(0, 0, 0, 0),		//Из какой части изображения берем
	size_t its = 0,													//В какую часть массива пихаем
	const float multiplier = 1.f / 255
) {
	if (roi.width == 0 && roi.height == 0)
		roi = cv::Rect(0, 0, mat.cols, mat.rows);

	//добавить очередной i-й кусочек входного изображения на its-ое место в батч
	for (int cm = 0; cm < mat.channels(); cm++)
		for (int im = 0; im < roi.height; im++)
			for (int jm = 0; jm < roi.width; jm++)
			{
				uchar* p = &mat.data[(((im + roi.y) * mat.cols + jm + roi.x) * mat.channels() + cm) * mat.elemSize1()];
				if (mat.depth() == CV_8U)
					data[its] = static_cast<mx_float>((*((uchar*)p)) * multiplier);
				if (mat.depth() == CV_32F)
					data[its] = static_cast<mx_float>((*((float*)p)) * multiplier);
				its++;
			}
}

inline void MxFloatArrToCVMat(
	const mx_float * data,									//отсюда без проверки границ
	cv::Mat &mat,														//сюда
	size_t its = 0,													//Из какой части массива берем
	cv::Rect roi = cv::Rect(0, 0, 0, 0),		//В какую часть изображения помещаем
	const float multiplier = 255.f
) {
	if (roi.width == 0 && roi.height == 0)
		roi = cv::Rect(0, 0, mat.cols, mat.rows);

	for (int cm = 0; cm < mat.channels(); cm++)
		for (int im = 0; im < roi.height; im++)
			for (int jm = 0; jm < roi.width; jm++)
			{
				if (mat.depth() == CV_8U)
					/**((unsigned char*)&*/mat.data[(((im + roi.y) * mat.cols + jm + roi.x) * mat.channels() + cm) * sizeof(unsigned char)] = static_cast<unsigned char>(data[its] * multiplier);
				if (mat.depth() == CV_32F)
					*((float*)&mat.data[(((im + roi.y) * mat.cols + jm + roi.x) * mat.channels() + cm) * sizeof(float)]) = static_cast<float>(data[its] * multiplier);
				its++;
			}
}

////convert cv::Mat to mxnet::cpp::NDArray format
////!!!Передевать NDArray параметром, если его размеры удовлетворяют, то не производить выделения памяти
////!!!Если порядок совпадает, то memcpy
//static mxnet::cpp::NDArray CVMatToMXNDArray(const cv::Mat &mat, const mxnet::cpp::Context &context)
//{
//	//if (nd_array.GetShape() )
//	mxnet::cpp::NDArray nd_array(mxnet::cpp::Shape(1, mat.channels(), mat.cols, mat.rows), context, false);
//	std::vector<mx_float> data;
//	data.reserve(1 * mat.channels() * mat.rows * mat.cols);
//
//	for (int c = 0; c < mat.channels(); c++)
//		for (int i = 0; i < mat.rows; i++) 
//			for (int j = 0; j < mat.cols; j++) 
//			{
//				data.emplace_back(static_cast<mx_float>(mat.data[(i * mat.cols + j) * mat.channels() + c]));
//			}
//
//	nd_array.SyncCopyFromCPU(data.data(), 1 * mat.channels() * mat.rows * mat.cols);
//	mxnet::cpp::NDArray::WaitAll();
//	return nd_array;
//}


///nd_array should already be not in GPU 
///result image type can be multichannel float mat CV_32FC(shape[1])
static cv::Mat MXNDArrayToCVMat(const mxnet::cpp::NDArray nd_array, mxnet::cpp::Shape shape = mxnet::cpp::Shape()/*, const mxnet::cpp::Context &context*/)
{
	if (shape == mxnet::cpp::Shape())
		shape = nd_array.GetShape();
	const mx_float* data = nd_array.GetData();
	cv::Mat result(shape[3], shape[2], CV_32FC(shape[1]));
	MxFloatArrToCVMat(data, result, 0, cv::Rect(0, 0, 0, 0), 1.f);
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


///Вывод одного изображения из батча
inline void ShowImageFromBatch(
	const std::string &win_name,
	const mxnet::cpp::NDArray &batch,
	const unsigned int width,
	const unsigned int height,
	const unsigned int channels,
	const bool colorize = true
){
	mxnet::cpp::Context context_cpu(mxnet::cpp::DeviceType::kCPU, 0);
	auto nda = batch.Copy(context_cpu);
	mxnet::cpp::NDArray::WaitAll();
	cv::Mat mat = MXNDArrayToCVMat(nda, mxnet::cpp::Shape(1, channels, width, height));
	cv::Mat temp;
	if (channels == 1)
	{
		if (!colorize)
		{
			temp = cv::Mat(mat.rows, mat.cols, CV_8U);
			cv::normalize(mat, temp, 0, 255, cv::NORM_MINMAX, CV_8U);
		}
		else {
			temp = Colorize(mat);
		}
	}
	else {
		temp = cv::Mat(mat.rows, mat.cols, CV_8UC3);
		cv::normalize(mat, temp, 0, 255, cv::NORM_MINMAX, CV_8UC3);
	}
	cv::imshow(win_name, temp);
}

#endif //USE_MXNET

#endif