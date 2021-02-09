#include <TFacesDataset.h>

#include <RandomWarpedImage.h>

//Копия params, но с отключенным Warp
TDatasetProperties GetNoWarpParams(const TDatasetProperties &properties)
{
	TDatasetProperties result = properties;
	result.Warp = false;
	return result;
}

TFacesDataset :: TFacesDataset(const TFacesDatasetProperties &faces_dataset_properties)
	:	TDataset(GetNoWarpParams(faces_dataset_properties.DatasetProperties))
{
	FacesDatasetProperties = faces_dataset_properties; 
	if (FacesDatasetProperties.DatasetProperties.UseMultiThreading)
	{
		//ThreadPool = new dlib::thread_pool(std::thread::hardware_concurrency()); //>parent
		FaceDetectors.resize(std::thread::hardware_concurrency());
		for (unsigned long i = 0; i < std::thread::hardware_concurrency(); i++)
			FaceDetectors[i] = new TFaceDetector(FacesDatasetProperties.PoseModelFilename);
		FreeDetectorIdx = 0;
	} else {
		FaceDetectors.resize(1);
		FaceDetectors[0] = new TFaceDetector(FacesDatasetProperties.PoseModelFilename);
	}
}

TFacesDataset :: ~TFacesDataset()
{
	if (FacesDatasetProperties.DatasetProperties.UseMultiThreading)
	{
		for (unsigned long i = 0; i < std::thread::hardware_concurrency(); i++)
			delete FaceDetectors[i];
			//delete ThreadPool;	//>parent
	} else {
		delete FaceDetectors[0]; 
	}
}

//Получить объект из выборки по индексу
cv::Mat TFacesDataset :: GetSampleCVMat(const unsigned long sample_idx)
{
	cv::Mat result;
	dlib::toMat(GetSampleDLibMatrix(sample_idx)).copyTo(result); 
	return result;
}

//Получить объект из выборки по индексу
dlib::matrix<unsigned char> TFacesDataset :: GetSampleDLibMatrix(const unsigned long sample_idx)
{
	dlib::matrix<unsigned char> //temp, 
															result(FacesDatasetProperties.DatasetProperties.ImgSize.height, FacesDatasetProperties.DatasetProperties.ImgSize.width);
	cv::Mat gray_img_buf = TDataset::GetSampleCVMat(sample_idx);
	// //Преобразуем ее в dlib::matrix
	//CVMatToDlibMatrix8U(gray_img_buf, temp);
	std::vector<dlib::rectangle> face_rects;
	dlib::array<dlib::matrix<unsigned char> > faces;
	//Detect faces 
	if (FacesDatasetProperties.DatasetProperties.UseMultiThreading)
	{
		unsigned long idx;
		DetectMutex.lock();
		FreeDetectorIdx++;
		FreeDetectorIdx = FreeDetectorIdx % std::thread::hardware_concurrency();
		idx = FreeDetectorIdx;
		DetectMutex.unlock();
		face_rects = FaceDetectors[idx]->Detect(gray_img_buf);
		FaceDetectors[idx]->ExtractFaces(gray_img_buf, face_rects, faces);
	} else {
		face_rects = FaceDetectors[0]->Detect(gray_img_buf);
		FaceDetectors[0]->ExtractFaces(gray_img_buf, face_rects, faces);
	};
	
	//dlib::interpolate_bilinear b;
	if (faces.size() == 1) resize_image(faces[0], result/*, b*/);	// > 1 - выбрать с макс площадью, 0 - exception
	if (faces.size() < 1) throw TException("Error TDataset :: GetSample: face not found on image from path: "+ LabeledSamplePaths[sample_idx].SamplePath);
	if (faces.size() > 1)
	{
		long max_sq = 0;
		for (unsigned long i = 0; i < faces.size(); i++)
		{
			if (i == 0 || faces[i].nc() * faces[i].nr() > max_sq)
			{
				max_sq = faces[i].nc() * faces[i].nr();
				resize_image(faces[i], result/*, b*/);
			}
		}
	}
	cv::Mat result_mat = dlib::toMat(result);
	if (FacesDatasetProperties.DatasetProperties.Warp)
		result_mat = GetRandomWarpedImage(
			result_mat,
			FacesDatasetProperties.DatasetProperties.WidthVariationFactor,
			FacesDatasetProperties.DatasetProperties.HeightVariationFactor,
			FacesDatasetProperties.DatasetProperties.BrightnessVariationFactor,
			FacesDatasetProperties.DatasetProperties.FlipVertical,
			FacesDatasetProperties.DatasetProperties.FlipHorizontal,
			false,
			FacesDatasetProperties.DatasetProperties.AddTerminator,
			FacesDatasetProperties.DatasetProperties.TerminatorAlphaMax,
			FacesDatasetProperties.DatasetProperties.MaxMotionBlurLen
		);
	CVMatToDlibMatrix8U(result_mat, result);

	return result;
}