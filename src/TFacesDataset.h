#ifndef TFACES_DATASET_H
#define TFACES_DATASET_H

#include <TDataset.h>
#include <TFaceDetector.h>

///Параметры датасета для работы с обучающей выборкой
/*typedef*/ struct TFacesDatasetProperties {
	TDatasetProperties DatasetProperties;
	std::string PoseModelFilename;
};
		
///Параметры датасета для работы с обучающей выборкой по умолчанию
static const TFacesDatasetProperties FACES_DATASET_PROPERTIES_DEFAULTS = 
{
	DATASET_PROPERTIES_DEFAULTS,
	""
};

///
class TFacesDataset : public TDataset{
protected:
	TFacesDatasetProperties FacesDatasetProperties;
	std::vector <TFaceDetector *> FaceDetectors;						///Детекторы лиц по одному на поток
	unsigned long FreeDetectorIdx;													///Способ распределить детектор по потокам


public:
	///Получить объект из выборки по индексу
	virtual cv::Mat GetSampleCVMat(const unsigned long sample_idx);


	TFacesDataset(const TFacesDatasetProperties &faces_dataset_properties);
	virtual ~TFacesDataset();

	///Получить объект из выборки по индексу
	virtual dlib::matrix<unsigned char> GetSampleDLibMatrix(const unsigned long sample_idx);
};

#endif