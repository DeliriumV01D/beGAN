#pragma once

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
#include "resnet_mx.h"

DISABLE_WARNING_POP

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include <TRandomInt.h>

struct TMNISTDatasetProperties
{
	std::string ImagePath,
		LabelPath;
	cv::Size ImgSize;
};

///Тестовый датасет для быстрой проверки 
class TMNISTDataset {
protected:
	const int NumberOfImages = 60000;
	const int NumberOfClasses = 10;
	const int NumberOfColumns = 28;
	const int NumberOfRows = 28;
	std::vector<char> Labels;
	TMNISTDatasetProperties DatasetProperties;
public:
	TMNISTDataset(const TMNISTDatasetProperties& mnist_dataset_properties)
	{
		DatasetProperties = mnist_dataset_properties;
		Labels.resize(NumberOfImages);

		std::filebuf buffer;
		std::istream stream(&buffer);
		//Читаем label-файл целиком в память
		if (buffer.open(DatasetProperties.LabelPath, std::ios::in | std::ios::binary | std::ios::ate))
		{
			stream.seekg(8);
			for (int i = 0; i < NumberOfImages; i++)
			{
				stream.read(&Labels[i], 1);
			};
		}
		stream.clear();
		buffer.close();
	}

	virtual ~TMNISTDataset() {};

	//По-хорошему, нужен случайный сэмпл
	virtual void GetSample(cv::Mat& sample, unsigned long& label)
	{
		int idx = RandomInt() % this->Size();
		cv::Mat temp(NumberOfRows, NumberOfColumns, CV_8UC1);
		std::filebuf buffer;
		std::istream stream(&buffer);
		if (buffer.open(DatasetProperties.ImagePath, std::ios::in | std::ios::binary | std::ios::ate))
		{
			stream.seekg(16 + idx * NumberOfRows * NumberOfColumns);
			for (int i = 0; i < NumberOfRows; i++)
				for (int j = 0; j < NumberOfColumns; j++)
				{
					unsigned char b;
					stream >> b;
					temp.at<unsigned char>(i, j) = b;
				};
		}
		stream.clear();
		buffer.close();

		cv::resize(temp, sample, DatasetProperties.ImgSize);
		label = Labels[idx];
	}


	void GetRandomSampleBatch(
		std::vector<mx_float>& batch_samples,
		std::vector<mx_float>& batch_labels,
		const size_t batch_size,
		const mxnet::cpp::Context& context
	) {
		size_t samples_size = batch_size * 1/*mat.channels()*/ * DatasetProperties.ImgSize.height/*rows*/ * DatasetProperties.ImgSize.width/*cols*/,
			labels_size = batch_size;

		if (batch_samples.size() != samples_size)
			batch_samples.resize(samples_size);
		if (batch_labels.size() != labels_size)
			batch_labels.resize(labels_size);

		cv::Mat sample;
		unsigned long label;
		size_t it = 0;
		for (unsigned long i = 0; i < batch_size; i++)
		{
			GetSample(sample, label);
			cv::resize(sample, sample, DatasetProperties.ImgSize);		//!!!делать resize внутри (перед аугментацией)
			cv::Scalar mean, stddev;
			meanStdDev(sample, mean, stddev);

			size_t it = i * 1/*mat.channels()*/ * DatasetProperties.ImgSize.height/*rows*/ * DatasetProperties.ImgSize.width/*cols*/;
			for (int cm = 0; cm < sample.channels(); cm++)
				for (int im = 0; im < sample.rows; im++)
					for (int jm = 0; jm < sample.cols; jm++)
					{
						batch_samples[it] = (static_cast<float>(sample.data[(im * sample.rows + jm) * sample.channels() + cm]) / 255 - (float)(mean[0]) / 255) / ((float)(stddev[0]) / 255);
						it++;
					}
			batch_labels[i] = (mx_float)label;
		}
	}

	virtual unsigned long Size()
	{
		return NumberOfImages;
	}

	virtual unsigned long ClassNumber()
	{
		return NumberOfClasses;
	}
};