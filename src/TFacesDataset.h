#ifndef TFACES_DATASET_H
#define TFACES_DATASET_H

#include <TDataset.h>
#include <TFaceDetector.h>

///��������� �������� ��� ������ � ��������� ��������
/*typedef*/ struct TFacesDatasetProperties {
	TDatasetProperties DatasetProperties;
	std::string PoseModelFilename;
};
		
///��������� �������� ��� ������ � ��������� �������� �� ���������
static const TFacesDatasetProperties FACES_DATASET_PROPERTIES_DEFAULTS = 
{
	DATASET_PROPERTIES_DEFAULTS,
	""
};

///
class TFacesDataset : public TDataset{
protected:
	TFacesDatasetProperties FacesDatasetProperties;
	std::vector <TFaceDetector *> FaceDetectors;						///��������� ��� �� ������ �� �����
	unsigned long FreeDetectorIdx;													///������ ������������ �������� �� �������


public:
	///�������� ������ �� ������� �� �������
	virtual cv::Mat GetSampleCVMat(const unsigned long sample_idx);


	TFacesDataset(const TFacesDatasetProperties &faces_dataset_properties);
	virtual ~TFacesDataset();

	///�������� ������ �� ������� �� �������
	virtual dlib::matrix<unsigned char> GetSampleDLibMatrix(const unsigned long sample_idx);
};

#endif