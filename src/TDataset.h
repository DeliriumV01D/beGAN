#ifndef TDATASET_H
#define TDATASET_H

#include <CommonDefinitions.h>
#include <TRandomInt.h>
#include <TIndexedObjects.h>
#include <TData.h>
#include <string>
#include <mutex>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>

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

///Вспомогательная структура для хранения адресов элементов выборки и их меток
class TLabeledSamplePath {
public:
	std::string SamplePath;
	unsigned long Label;

	TLabeledSamplePath(const std::string sample_path, const unsigned long label)
	{
		SamplePath = sample_path;
		Label = label;
	};
};

///Параметры датасета для работы с обучающей выборкой
/*typedef*/ struct TDatasetProperties {
 	std::string Dir;
	cv::Size ImgSize;
	bool	UseMultiThreading,
				OneThreadReading,
				Warp,
				FlipVertical,
				FlipHorizontal,
				AddTerminator;
	double TerminatorAlphaMax;
	float WidthVariationFactor,
				HeightVariationFactor,
				BrightnessVariationFactor;
	int MaxMotionBlurLen;
};

///Параметры датасета для работы с обучающей выборкой по умолчанию
static const TDatasetProperties DATASET_PROPERTIES_DEFAULTS = 
{
	"",
	cv::Size(0, 0),
	true,
	false,
	true,
	true,
	false,
	true,
	0.2,
	0.05,
	0.05,
	0.1,
	5
};

///Датасет для работы с обучающей выборкой
class TDataset {
protected:
	TDatasetProperties DatasetProperties;
	TIndexedObjects <std::string> Labels;										///Метки классов - имена подкаталогов, текст из файлов
	std::vector <TLabeledSamplePath> LabeledSamplePaths;		///Адреса элементов обучающей выборки и метки классов
	dlib::thread_pool * ThreadPool;
	std::mutex	ReadMutex,																	///На случай one_thread_reading закроет imread
							DetectMutex;
public:
	TDataset(const TDatasetProperties &dataset_properties);
	virtual ~TDataset();

	virtual unsigned long Size();
	virtual unsigned long ClassNumber();

	///Работаем в предположении, что все изображения одного объекта лежат в одном подкаталоге
	virtual std::string GetLabelFromPath(const std::string &path);
	///Получить объект из выборки по индексу
	virtual cv::Mat GetSampleCVMat(const unsigned long sample_idx);
	///Получить объект из выборки по индексу
	virtual dlib::matrix<unsigned char> GetSampleDLibMatrix(const unsigned long sample_idx);
	///Получить метку по индексу объекта
	virtual std::string GetLabel(const unsigned long sample_idx);
	///Получить метку по индексу метки
	virtual std::string GetLabelByIdx(const unsigned long label_idx);
	///Получить индекс метки из выборки по индексу
	virtual unsigned long GetLabelIdx(const unsigned long sample_idx);
	///Сформировать вход нейросетки по двум изображениям(одинакового размера!)
	virtual dlib::matrix<unsigned char> MakeInputSamplePair(dlib::matrix<unsigned char> * img1, dlib::matrix<unsigned char> * img2);

	///Получить пару изображений positive == true одного объекта, false - разных объектов и соответствующую метку
	virtual void GetInputSamplePair(bool positive, dlib::matrix<unsigned char> &sample_pair, unsigned long &label);
	///Получить пакет пар изображений и соответвующие метки
	virtual void GetInputSamplePairBatch(
		std::vector<dlib::matrix<unsigned char>> &batch_sample_pairs, 
		std::vector<unsigned long> &batch_labels,
		const size_t batch_size
	);

	///Случайный объект из датасета и соответствующая ему метка	 cv::Mat - dlib::matrix переписать через шаблон?
	virtual void GetRandomSample(dlib::matrix<unsigned char> &sample, unsigned long &label);
	///Случайный объект из датасета и соответствующая ему метка  cv::Mat - dlib::matrix переписать через шаблон?
	virtual void GetRandomSample(cv::Mat &sample, unsigned long &label);
	///Получить пакет изображений и соответствующие им метки  cv::Mat - dlib::matrix переписать через шаблон?
	virtual void GetRandomSampleBatch(
		std::vector<dlib::matrix<unsigned char>> &batch_samples, 
		std::vector<unsigned long> &batch_labels,
		const size_t batch_size
	);
	/////Получить пакет изображений и соответствующие им метки  cv::Mat - dlib::matrix переписать через шаблон?
	//virtual void GetRandomSampleBatch(
	//	cv::Mat &data, 
	//	cv::Mat &classes,
	//	const size_t batch_size
	//);

	#ifdef USE_MXNET
	void GetRandomSampleBatch(
		std::vector<mx_float> &batch_samples,
		std::vector<mx_float> &batch_labels,
		const size_t batch_size
	);
	#endif

};

#endif