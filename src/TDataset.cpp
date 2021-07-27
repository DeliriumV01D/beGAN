#include <TDataset.h>
#include <list>
#include <vector>
#include <time.h>
#include <WriteToLog.h>
#include <RandomWarpedImage.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <dlib/matrix.h>
#include <dlib/threads.h>

const std::string LABEL_FILE_NAME = "label.txt";

//Формируем вектор путей файлов изображений и соответствующих им меток
TDataset :: TDataset(const TDatasetProperties &dataset_properties)
{
	DatasetProperties = dataset_properties;
	TRandomInt * ri = &(TRandomInt::Instance());
	ri->Initialize(time(0));
	if (DatasetProperties.UseMultiThreading)
	{
		ThreadPool = new dlib::thread_pool(std::thread::hardware_concurrency());
	}

	unsigned long label;
	std::list <std::string> :: const_iterator	current_position;  
	std::list <std::string> data_list = getFilesNames(DatasetProperties.Dir, "*.bmp");
	data_list.splice(data_list.end(), getFilesNames(DatasetProperties.Dir, "*.jpg"));
	data_list.splice(data_list.end(), getFilesNames(DatasetProperties.Dir, "*.jpeg"));

	for (	current_position = data_list.begin(); 
				current_position != data_list.end(); 
				++current_position
			)
	{
		label = Labels.GetIdxByObj(GetLabelFromPath(*current_position));
		LabeledSamplePaths.push_back(TLabeledSamplePath(*current_position, label));
	};
}

TDataset :: ~TDataset()
{
	if (DatasetProperties.UseMultiThreading)
	{
		delete ThreadPool;
	}
}

//Работаем в предположении, что все изображения одного объекта лежат в одном подкаталоге
std::string TDataset :: GetLabelFromPath(const std::string &path)
{
	std::string s;
	size_t l = path.find_last_of("/\\");
	if (ReadStringFromFile(path.substr(0, l + 1) + LABEL_FILE_NAME, s))
		return s;
	return path.substr(0, l + 1);
}

unsigned long TDataset :: Size()
{
	return LabeledSamplePaths.size();
}

unsigned long TDataset :: ClassNumber()
{
	return Labels.Size();
}

//Получить объект из выборки по индексу
cv::Mat TDataset :: GetSampleCVMat(const unsigned long sample_idx)
{
	if (sample_idx < 0 || sample_idx >= this->Size()) throw TException("Error TDataset :: GetSample: sample index is incorrect");

	cv::Mat result;
	//Читаем исходную картинку
	bool lock = DatasetProperties.UseMultiThreading && DatasetProperties.OneThreadReading;
	if (lock) ReadMutex.lock();
	try {
		result = cv::imread(LabeledSamplePaths[sample_idx].SamplePath.c_str(), cv::ImreadModes::IMREAD_GRAYSCALE);
	} catch (std::exception &e) {
		if (lock) ReadMutex.unlock();

		TException * E = dynamic_cast<TException *>(&e);
		if (E) throw (*E);
		else throw e;
	};
	if (lock) ReadMutex.unlock();
  if (!result.data) throw TException("Error TDataset :: GetSample: can not read image from path: " + LabeledSamplePaths[sample_idx].SamplePath);

	if (DatasetProperties.Warp) 
		result = GetRandomWarpedImage(
			result, 
			DatasetProperties.WidthVariationFactor, 
			DatasetProperties.HeightVariationFactor, 
			DatasetProperties.BrightnessVariationFactor,
			DatasetProperties.FlipVertical,
			DatasetProperties.FlipHorizontal,
			false,
			DatasetProperties.AddTerminator,
			DatasetProperties.TerminatorAlphaMax,
			DatasetProperties.MaxMotionBlurLen
		);
	return result;
}

//Получить объект из выборки по индексу
dlib::matrix<unsigned char> TDataset :: GetSampleDLibMatrix(const unsigned long sample_idx)
{
	dlib::matrix<unsigned char> temp, 
															result(DatasetProperties.ImgSize.height, DatasetProperties.ImgSize.width);
	cv::Mat gray_img_buf = GetSampleCVMat(sample_idx);
	//Преобразуем ее в dlib::matrix
	CVMatToDlibMatrix8U(gray_img_buf, temp);
	resize_image(temp, result);
	return result;
}

//Получить метку по индексу объекта
std::string TDataset :: GetLabel(const unsigned long sample_idx)
{
	return GetLabelByIdx(LabeledSamplePaths[sample_idx].Label);
}

///Получить метку по индексу метки
std::string TDataset :: GetLabelByIdx(const unsigned long label_idx)
{
	return Labels.GetObjByIdx(label_idx);
}

//Получить индекс метки из выборки по индексу
unsigned long TDataset :: GetLabelIdx(const unsigned long sample_idx)
{
	return LabeledSamplePaths[sample_idx].Label;
}

//Сформировать вход нейросетки по двум изображениям(одинакового размера!)
dlib::matrix<unsigned char> TDataset :: MakeInputSamplePair(dlib::matrix<unsigned char> * img1, dlib::matrix<unsigned char> * img2)
{
	dlib::matrix<unsigned char> result;
	result.set_size(img1->nr() * 2, img1->nc());
	dlib::set_subm(result, dlib::range(0, img1->nr() - 1), dlib::range(0, img1->nc() - 1)) = *img1;	//rectangle(0,0,1,2)
	dlib::set_subm(result, dlib::range(img2->nr(), img2->nr() * 2 - 1), dlib::range(0, img2->nc() - 1)) = *img2;	//rectangle(0,0,1,2)
	return result;
}

///Получить пару изображений positive == true одного объекта, false - разных объектов и соответствующую метку
void TDataset :: GetInputSamplePair(bool positive, dlib::matrix<unsigned char> &sample_pair, unsigned long &label)
{
	unsigned long idx, jdx;
	dlib::matrix<unsigned char> img1, img2;
	const unsigned long locality = 30,
											attempts = 200;
	if (this->Size() == 0) return;
	bool found = false;
	
	do {
		//Берем случайный индекс
		idx = RandomInt() % this->Size();
		if (positive){ //если positive, то ищем в окрестности, пока не будет совпадения label
			for (unsigned long i = 0; i < attempts; i++)
			{
				jdx = idx - locality/2 + RandomInt() % locality;
				if (jdx < 0 || jdx > this->Size() - 1) continue;
				if (GetLabelIdx(idx) == GetLabelIdx(jdx))
				{
					found = true;
					break;
				};
			}
		} else {//if positive
			//Если negative, то просто повторяем поиск случайного индекса, пока не наткнемся на label, отличающийся
			for (unsigned long i = 0; i < attempts; i++)
			{
				jdx = RandomInt() % this->Size();
				if (GetLabelIdx(idx) != GetLabelIdx(jdx)) 
				{
					found = true;
					break;
				};
			}
		};
		try {
			img1 = GetSampleDLibMatrix(idx);
			img2 = GetSampleDLibMatrix(jdx);
		} catch (std::exception &e) {
			found = false;
			TException * E = dynamic_cast<TException *>(&e);
			if (E)
				std::cout<<E->what()<<std::endl;
			else
				std::cout<<e.what()<<std::endl;
		};
	} while (!found);
	sample_pair = MakeInputSamplePair(&img1, &img2);
	if (positive) label = 1; else label = 0;
}

//class TF {
//protected:
//	TDataset * Dataset; 
//	bool Positive; 
//public:
//	dlib::matrix<unsigned char> SamplePair;
//	unsigned long Label;
//
//	TF(TDataset * dataset, bool positive)
//	{
//		Dataset = dataset;
//		Positive = positive;
//	};
//	void operator () ()
//	{
//		Dataset->GetInputSamplePair(Positive, SamplePair, Label);
//	};
//};

struct TR {
	TDataset * Dataset; 
	bool Positive; 
	dlib::matrix<unsigned char> SamplePair;
	unsigned long Label;
};

//Получить пакет пар изображений и соответвующие метки
void TDataset :: GetInputSamplePairBatch(
	std::vector<dlib::matrix<unsigned char>> &batch_sample_pairs, 
	std::vector<unsigned long> &batch_labels,
	const size_t batch_size
){
	bool positive = true;
	dlib::matrix<unsigned char> sample_pair;
	unsigned long label;

	batch_sample_pairs.clear();
	batch_labels.clear();
	if (DatasetProperties.UseMultiThreading)
	{
		std::vector <TR> trv;
		std::vector<dlib::future<TR>> fv(batch_size);
		for (unsigned long i = 0; i < batch_size; i++)
		{
			trv.push_back(TR() = {this, positive});
	    fv[i] = trv[i];
			//Можно, конечно же обойтись и без всяких лямбд
			ThreadPool->add_task_by_value([](TR &val){val.Dataset->GetInputSamplePair(val.Positive, val.SamplePair, val.Label);}, fv[i]);
			positive = !positive;	//Чередуем положительные и отрицательные примеры
		};
		//Каждому потоку нужен свой экземпляр детектора, потому что может быть разного размера и тд.
		ThreadPool->wait_for_all_tasks();
		for (unsigned long i = 0; i < batch_size; i++)
		{
			batch_sample_pairs.push_back(fv[i].get().SamplePair);
			batch_labels.push_back(fv[i].get().Label);
			//batch_sample_pairs.push_back(tfv[i].SamplePair);
			//batch_labels.push_back(tfv[i].Label);
		};
	} else {
		for (unsigned long i = 0; i < batch_size; i++)
		{
			GetInputSamplePair(positive, sample_pair, label);
			batch_sample_pairs.push_back(sample_pair);
			batch_labels.push_back(label);
			positive = !positive;	//Чередуем положительные и отрицательные примеры
		};
	}
}

//Случайный объект из датасета и соответствующая ему метка
void TDataset :: GetRandomSample(dlib::matrix<unsigned char> &sample, unsigned long &label)
{
	unsigned long idx;
	if (this->Size() == 0) return;
	bool found = false;
	
	do {
		//Берем случайный индекс
		idx = RandomInt() % this->Size();
		try {
			sample = GetSampleDLibMatrix(idx);
			label = GetLabelIdx(idx);
			found = true;
		} catch (std::exception &e) {
			found = false;
			//TException * E = dynamic_cast<TException *>(&e);
			//if (E)
			//	std::cout<<E->what()<<std::endl;
			//else
			//	std::cout<<e.what()<<std::endl;
		};
	} while (!found);
}

//Случайный объект из датасета и соответствующая ему метка
void TDataset :: GetRandomSample(cv::Mat &sample, unsigned long &label)
{
	unsigned long idx;
	if (this->Size() == 0) return;
	bool found = false;
	
	do {
		//Берем случайный индекс
		idx = RandomInt() % this->Size();
		try {
			sample = GetSampleCVMat(idx);
			label = GetLabelIdx(idx);
			found = true;
		} catch (std::exception &e) {
			found = false;
		};
	} while (!found);
}


//Получить пакет изображений и соответствующие им метки  cv::Mat - dlib::matrix переписать через шаблон?
void TDataset :: GetRandomSampleBatch(
	std::vector<dlib::matrix<unsigned char>> &batch_samples, 
	std::vector<unsigned long> &batch_labels,
	const size_t batch_size
) {
	dlib::matrix<unsigned char> sample;
	unsigned long label;

	struct TS {
		TDataset * Dataset;
		dlib::matrix<unsigned char> Sample;
		unsigned long Label;
	};

	batch_samples.clear();
	batch_labels.clear();
	if (DatasetProperties.UseMultiThreading)
	{
		std::vector <TS> tsv;
		std::vector<dlib::future<TS>> fv(batch_size);
		for (unsigned long i = 0; i < batch_size; i++)
		{
			tsv.push_back(TS() = {this});
	    fv[i] = tsv[i];
			ThreadPool->add_task_by_value([](TS &val){val.Dataset->GetRandomSample(val.Sample, val.Label);}, fv[i]);
		};
		//Каждому потоку нужен свой экземпляр детектора, потому что может быть разного размера и тд.
		ThreadPool->wait_for_all_tasks();
		for (unsigned long i = 0; i < batch_size; i++)
		{
			batch_samples.push_back(fv[i].get().Sample);
			batch_labels.push_back(fv[i].get().Label);
		};
	} else {
		for (unsigned long i = 0; i < batch_size; i++)
		{
			GetRandomSample(sample, label);
			batch_samples.push_back(sample);
			batch_labels.push_back(label);
		};
	}
}


#ifdef USE_MXNET
///
void TDataset :: GetRandomSampleBatch(
	std::vector<mx_float> &batch_samples,
	std::vector<mx_float> &batch_labels,
	const size_t batch_size
) {
	struct TS {
		TDataset* Dataset;
		cv::Mat Sample;
		unsigned long Label;
	};

	size_t samples_size = batch_size * 1/*mat.channels()*/ * DatasetProperties.ImgSize.height/*rows*/ * DatasetProperties.ImgSize.width/*cols*/,
												labels_size = batch_size;

	if (batch_samples.size() != samples_size)
		batch_samples.resize(samples_size);
	if (batch_labels.size() != labels_size)
		batch_labels.resize(labels_size);

	if (DatasetProperties.UseMultiThreading)
	{
		for (unsigned long i = 0; i < batch_size; i++)
		{
			ThreadPool->add_task_by_value(
				[this, i, &batch_samples, &batch_labels]() {		//???Нужно передать указатели на позицию в массиве и их итерировать, чтобы оптимизировать кэш
					cv::Mat sample;
					unsigned long label;
					GetRandomSample(sample, label);
					cv::resize(sample, sample, DatasetProperties.ImgSize);	//!!!делать resize внутри (перед аугментацией или после)
					//cv::Scalar mean, stddev;
					//cv::meanStdDev(sample, mean, stddev);

					size_t it = i * 1/*mat.channels()*/ * DatasetProperties.ImgSize.height/*rows*/ * DatasetProperties.ImgSize.width/*cols*/;
					CVMatToMxFloatArr(sample, &batch_samples[0], cv::Rect(0, 0, 0, 0), it, 1.f / 255);
					batch_labels[i] = label;
				}
			);
		};
		//Каждому потоку нужен свой экземпляр детектора, потому что может быть разного размера и тд.
		ThreadPool->wait_for_all_tasks();

	}	else {

		cv::Mat sample;
		unsigned long label;
		size_t it = 0;
		for (unsigned long i = 0; i < batch_size; i++)
		{
			GetRandomSample(sample, label);
			cv::resize(sample, sample, DatasetProperties.ImgSize);		//!!!делать resize внутри (перед аугментацией)
			//cv::Scalar mean, stddev;
			//cv::meanStdDev(sample, mean, stddev);

			size_t it = i * 1/*mat.channels()*/ * DatasetProperties.ImgSize.height/*rows*/ * DatasetProperties.ImgSize.width/*cols*/;
			CVMatToMxFloatArr(sample,	&batch_samples[0], cv::Rect(0, 0, 0, 0), it, 1.f / 255);
			batch_labels[i] = label;
		};
	}
}

#endif