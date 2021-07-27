#pragma once

#include <string>
#include <map>
#include <filesystem>
#include <memory>
#include <thread>
#include <algorithm>
#include <unordered_set>


#include "TRandomDouble.h"
#include "CommonDefinitions.h"
#include "TData.h"
#include "model_executor.h"


DISABLE_WARNING_PUSH

#if defined(_MSC_VER)
DISABLE_WARNING(4267)
DISABLE_WARNING(4305)
DISABLE_WARNING(4251)
DISABLE_WARNING(4244)
#endif

#include "mxnet-cpp/MxNetCpp.h"
#include "resnet_mx.h"
#include "TMxDNNScheduler.h"
#include "mxnet-cpp/metric.h"

DISABLE_WARNING_POP


enum class TMxDNNTrainerOptimizer { ADAM, SGD };


inline std::unordered_set<std::string> GetFilteredArgumentSet(const std::vector<std::string> &arguments, const std::string containing)
{
	std::unordered_set<std::string> result;
	for (auto &it : arguments)
	{
		if (containing.empty() || it.find(containing) != std::string::npos)
			result.insert(it);
	}
	return result;
}

inline std::unordered_set<std::string> GetFilteredArgumentSet(const mxnet::cpp::Symbol &symbol, const std::string containing)
{
	std::vector<std::string> arguments = symbol.ListArguments();
	return GetFilteredArgumentSet(arguments, containing);
}

///Метрики стоит вынести в отдельный модуль
class NMAE : public mxnet::cpp::EvalMetric {
public:
	NMAE() : EvalMetric("nmae") {}

	void Update(mxnet::cpp::NDArray labels, mxnet::cpp::NDArray preds) override 
	{
		CheckLabelShapes(labels, preds);

		std::vector<mx_float> pred_data;
		preds.SyncCopyToCPU(&pred_data);
		std::vector<mx_float> label_data;
		labels.SyncCopyToCPU(&label_data);

		size_t len = preds.Size();
		mx_float sum = 0;
		for (size_t i = 0; i < len; ++i) 
		{
			sum += std::abs(pred_data[i]  - label_data[i]);
		}
		sum_metric += sum / len;
		++num_inst;
	}
};


struct TMxDNNTrainerProperties {
	ModelExecutorProperties ExecutorProperties;
	int 
			MaxEpoch,
			EpochStep,						//Здесь указывается число step итераций(батчей) после которого learning rate уменьшается в factor раз
			NBatchStatistics,			//Считаем метрики по этому числу батчей
			TestBatches,					//Количество батчей тестового датасета для тестирования один раз за эпоху
			OutputInfoTime;				//Периодичность вывода дежурной информации в секундах
	float StartLearningRate,
				FinalLearningRate,
				WeightDecay,
				Err;								//abs(1. - Accuracy) for CLASSIFIER, MAE for AUTOENCODER, SEGMENTER
	std::string StateFileName;
	TMxDNNTrainerOptimizer TrainerOptimizer;
};


template <typename DatasetType>
class TMxDNNTrainer : public ModelExecutor {
protected:
	TMxDNNTrainerProperties Properties;
	DatasetType * TrainDataset,
							* TestDataset;
	std::vector<std::string> ArgNames;
	std::unordered_set<size_t> NonUpdatableArguments;

	static mxnet::cpp::NDArray ResizeInput(mxnet::cpp::NDArray data, const mxnet::cpp::Shape new_shape, const int dataset_image_width, const int dataset_image_height);
	virtual void InitializeMetrics(std::unique_ptr<mxnet::cpp::EvalMetric> &metric1, std::unique_ptr<mxnet::cpp::EvalMetric> &metric2);
	virtual void OutputInfo(
		mxnet::cpp::EvalMetric * metric1, 
		mxnet::cpp::EvalMetric * metric2,
		const int epoch,
		const int iter,
		mxnet::cpp::Executor * exec
	);

	virtual bool TerminationConditions(const std::unique_ptr<mxnet::cpp::EvalMetric> &metric1);
public:
	TMxDNNTrainer(
		TMxDNNTrainerProperties &properties,
		DatasetType * train_dataset,
		DatasetType * test_dataset /* = (Dataset*)nullptr*/,
		mxnet::cpp::Symbol * net,
		mxnet::cpp::Context * context = nullptr
	);

	virtual ~TMxDNNTrainer(){};

	virtual void Train();

	void SetNonUpdatableArguments(const std::unordered_set<size_t> &na_arguments)
	{
		NonUpdatableArguments = na_arguments;
	}

	static std::list<std::pair<std::string, std::string>> ParseKeyValues(const std::string& s);
	virtual void InitializeState(std::unique_ptr<mxnet::cpp::Optimizer> &opt);
	virtual void SaveState(mxnet::cpp::Optimizer * opt, const int epoch);
	virtual void LoadState(std::unique_ptr<mxnet::cpp::Optimizer> &opt, int &epoch);
};



template <typename DatasetType>
TMxDNNTrainer<DatasetType> :: TMxDNNTrainer(
	TMxDNNTrainerProperties &properties,
	DatasetType * train_dataset,
	DatasetType * test_dataset /* = (Dataset*)nullptr*/,
	mxnet::cpp::Symbol * net,
	mxnet::cpp::Context * context /*= nullptr*/
) : ModelExecutor(properties.ExecutorProperties, net, context) {
	Properties = properties;
	TrainDataset = train_dataset;
	TestDataset = test_dataset;
}

template <typename DatasetType>
mxnet::cpp::NDArray TMxDNNTrainer<DatasetType> :: ResizeInput(mxnet::cpp::NDArray data, const mxnet::cpp::Shape new_shape, const int dataset_image_width, const int dataset_image_height)
{
	mxnet::cpp::NDArray pic = data.Reshape(mxnet::cpp::Shape(1, 1, dataset_image_width, dataset_image_height));
	mxnet::cpp::NDArray output;
	mxnet::cpp::Operator("_contrib_BilinearResize2D")
		.SetParam("height", new_shape[2])
		.SetParam("width", new_shape[3])
		(pic).Invoke(output);
	return output;
}

template <typename DatasetType>
void TMxDNNTrainer<DatasetType> :: InitializeMetrics(std::unique_ptr<mxnet::cpp::EvalMetric> &metric1, std::unique_ptr<mxnet::cpp::EvalMetric> &metric2)
{
	if (Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::CLASSIFIER)
	{
		metric1 = std::make_unique<mxnet::cpp::Accuracy>();
		metric2 = std::make_unique<mxnet::cpp::LogLoss>();
	}
	if (Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::AUTOENCODER || Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::SEGMENTER)
	{
		metric1 = std::make_unique<mxnet::cpp::MAE>();
		metric2 = std::make_unique<mxnet::cpp::RMSE>();
	}
}



template <typename DatasetType>
void TMxDNNTrainer<DatasetType> :: OutputInfo(
	mxnet::cpp::EvalMetric * metric1, 
	mxnet::cpp::EvalMetric * metric2,
	const int epoch,
	const int iter,
	mxnet::cpp::Executor * exec
)
{
	//Вывод одного изображения из батча
	if (Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::AUTOENCODER || Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::SEGMENTER)
	{
		bool colorize = Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::SEGMENTER;
		ShowImageFromBatch("data1", ArgsMap["data"], Properties.ExecutorProperties.NetImageWidth, Properties.ExecutorProperties.NetImageHeight, Properties.ExecutorProperties.NetImageChannels, false);
		ShowImageFromBatch("label", ArgsMap["label"], Properties.ExecutorProperties.NetImageWidth, Properties.ExecutorProperties.NetImageHeight, Properties.ExecutorProperties.LabelChannels, colorize);
		ShowImageFromBatch("output", exec->outputs[0], Properties.ExecutorProperties.NetImageWidth, Properties.ExecutorProperties.NetImageHeight, Properties.ExecutorProperties.LabelChannels, colorize);
		cv::waitKey(1);
	}
	if (Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::CLASSIFIER)
		LG << "EPOCH: " << epoch << " ITER: " << iter << " Train Accuracy: " << metric1->Get() << " Train Loss: " << metric2->Get();
	if (Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::AUTOENCODER || Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::SEGMENTER)
		LG << "EPOCH: " << epoch << " ITER: " << iter << " MAE: " << metric1->Get() << " RMSE: " << metric2->Get();
}

template <typename DatasetType>
bool TMxDNNTrainer<DatasetType> :: TerminationConditions(const std::unique_ptr<mxnet::cpp::EvalMetric> &metric1)
{
	bool result = false;

	if (Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::CLASSIFIER)
		if (abs(metric1->Get() - 1.) < Properties.Err)		//Accuracy
			result = true;

	if (Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::AUTOENCODER || Properties.ExecutorProperties.ExecuteMode == DNNExecuteMode::SEGMENTER)
		if (metric1->Get() < Properties.Err)					//MAE
			result = true;

	return result;
}


///Обучение
template <typename DatasetType>
void TMxDNNTrainer<DatasetType> :: Train()
{
	auto logarithm = [](const double a, const double b) { return log(b) / log(a); };		//log b по основанию a

	std::unique_ptr<mxnet::cpp::Optimizer> opt;
	int start_epoch = 0;

	if (!std::filesystem::exists(Properties.StateFileName))
		InitializeState(opt);
	else
		LoadState(opt, start_epoch);

	const float factor = 0.1f;
	int step = Properties.EpochStep * TrainDataset->Size() / Properties.ExecutorProperties.BatchSize;
	//Здесь указывается число step итераций(батчей) после которого learning rate уменьшается в factor раз
	std::unique_ptr<TMxDNNScheduler> lr_sch(new TMxDNNScheduler(Properties.StartLearningRate, step, factor, Properties.FinalLearningRate, start_epoch * TrainDataset->Size() / Properties.ExecutorProperties.BatchSize));
	opt->SetLRScheduler(std::move(lr_sch));
	
	for (auto it : ArgsMap)
		std::cout << it.first << std::endl;
	std::cout << std::endl;

	// Create metrics
	std::unique_ptr<mxnet::cpp::EvalMetric> metric1,
																					metric2;
	InitializeMetrics(metric1, metric2);

	//ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
	std::vector<mx_float> batch_samples;
	std::vector<mx_float> batch_labels;
	std::chrono::steady_clock::time_point last_output_info = std::chrono::steady_clock::now();
	for (int epoch = start_epoch; epoch < std::min<float>((float)Properties.MaxEpoch, (float)(logarithm(1. / factor, Properties.StartLearningRate) - logarithm(1. / factor, Properties.FinalLearningRate) + 1) * Properties.EpochStep); ++epoch)
	{
		LG << "Epoch: " << epoch<< " of "
			<< std::min<float>((float)Properties.MaxEpoch, (float)(logarithm(1. / factor, Properties.StartLearningRate) - logarithm(1. / factor, Properties.FinalLearningRate) + 1) * Properties.EpochStep)
			<< "; " << opt->Serialize();

		int iter = 0;
		for (size_t dit = 0; dit <= TrainDataset->Size() / Properties.ExecutorProperties.BatchSize; dit++)		//!!!Ну приблизительно эпоха
		{
			TrainDataset->GetRandomSampleBatch(batch_samples, batch_labels, Properties.ExecutorProperties.BatchSize);

			SetArguments(batch_samples, batch_labels);
			
			Exec->Forward(true);
			Exec->Backward();

			for (size_t i = 0; i < ArgNames.size(); ++i)
			{
				if (ArgNames[i] == "data" || ArgNames[i] == "label") 
					continue;
				if (NonUpdatableArguments.find(i) == NonUpdatableArguments.end())
					opt->Update(static_cast<int>(i), Exec->arg_arrays[i], Exec->grad_arrays[i]);
			}

			mxnet::cpp::NDArray::WaitAll();

			auto output = Exec->outputs[0].Copy(*Context);
			mxnet::cpp::NDArray::WaitAll();

			if (dit %  Properties.NBatchStatistics == 0)			//!!!По-хорошему нужно что-то вроде окна
				metric1->Reset();
			metric1->Update(ArgsMap["label"], output);
			if (metric2 != nullptr)
			{
				if (dit %  Properties.NBatchStatistics == 0)			//!!!По-хорошему нужно что-то вроде окна
					metric2->Reset();
				metric2->Update(ArgsMap["label"], output);
			}

			if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - last_output_info).count() > Properties.OutputInfoTime)
			{
				OutputInfo(metric1.get(), metric2.get(), epoch, iter, Exec.get());
				last_output_info = std::chrono::steady_clock::now();
			}

			++iter;
		}

		SaveState(opt.get(), epoch + 1);

		if (TerminationConditions(metric1))
			break;

		//!!!Завести для инференса отдельную сеть, которую подгружать из файлика, чтобы исключить влияние на обучение
		//if (TestDataset != nullptr)				//!!!Нужно сохранять лог обучения с результатами по тестовому датасету
		//{
		//	if (metric2 != nullptr) 
		//		metric2->Reset();
		//	metric1->Reset();
		//	size_t test_batches = TestDataset->Size() / Properties.ExecutorProperties.BatchSize;
		//	if (Properties.TestBatches != 0 && Properties.TestBatches < test_batches)
		//		test_batches = Properties.TestBatches;
		//	for (size_t dit = 0; dit <= test_batches; dit++)		//!!!Тут переделать по порядку и с паддингом в конце
		//	{
		//		TestDataset->GetRandomSampleBatch(batch_samples, batch_labels, Properties.ExecutorProperties.BatchSize);
		//		auto output = Execute(batch_samples, std::vector<mx_float>(), false);

		//		metric1->Update(args_map["label"], output);
		//		if (metric2 != nullptr)
		//			metric2->Update(args_map["label"], output);
		//	}

		//	OutputInfo(metric1.get(), metric2.get(), epoch, iter, Exec.get());
		//}
	}			//for epoch
}

template <typename DatasetType>
void TMxDNNTrainer<DatasetType> :: InitializeState(std::unique_ptr<mxnet::cpp::Optimizer> &opt)
{
	InitializeNet(/*Exec,*/ArgNames);
	LG << "Initializing state with trainer properties";

	if (Properties.TrainerOptimizer == TMxDNNTrainerOptimizer::SGD)
	{
		LG << "optimizer: SGD";
		opt.reset(mxnet::cpp::OptimizerRegistry::Find("sgd"));
		opt->SetParam("lr", Properties.StartLearningRate)		//!!!Вот так параметры можно установить как достать смотреть ниже
			->SetParam("wd", Properties.WeightDecay)
			->SetParam("momentum", 0.99)
			->SetParam("rescale_grad", std::min(1., 1.0 / Properties.ExecutorProperties.BatchSize))
			->SetParam("lazy_update", false);
	}

	if (Properties.TrainerOptimizer == TMxDNNTrainerOptimizer::ADAM)
	{
		LG << "optimizer: ADAM";
		opt.reset(mxnet::cpp::OptimizerRegistry::Find("adam"));
		opt->SetParam("lr", Properties.StartLearningRate)		//!!!Вот так параметры можно установить как достать смотреть ниже
			->SetParam("rescale_grad", std::min(1., 1.0 / Properties.ExecutorProperties.BatchSize))
			->SetParam("beta1", 0.9)
			->SetParam("beta2", 0.999)
			->SetParam("epsilon", 1e-8)
			->SetParam("lazy_update", true)
			->SetParam("wd", Properties.WeightDecay);
	}
}

template <typename DatasetType>
void TMxDNNTrainer<DatasetType> :: SaveState(mxnet::cpp::Optimizer * opt, const int epoch)
{
	TMxDNNTrainer::SaveModelParameters(Properties.ExecutorProperties.ModelFileName, ArgsMap, ArgGradStore, GradReqType, AuxMap, *Context);

	LG << "Saving state to " << Properties.StateFileName;

	LG << "epoch = " << epoch << std::endl << opt->Serialize();

	WriteStringToFile(Properties.StateFileName, "epoch=" + std::to_string(epoch) + "\n" + opt->Serialize());
}

template <typename DatasetType>
void TMxDNNTrainer<DatasetType> :: LoadState(std::unique_ptr<mxnet::cpp::Optimizer> &opt, int &epoch)
{
	InitializeNet(/*Exec,*/ ArgNames);

	LG << "Loading state from " << Properties.StateFileName;

	std::string s;
	if (!ReadStringFromFile(Properties.StateFileName, s))
		std::runtime_error("TMxDNNTrainer :: LoadState error: can not read file " + Properties.StateFileName);

	LG << s;

	//Дальше их парсить и заполнять значения параметров
	auto kvl = ParseKeyValues(s);

	for (auto& it : kvl)
		if (it.first == "opt_type")
			opt.reset(mxnet::cpp::OptimizerRegistry::Find(it.second));
	for (auto& it : kvl)
	{
		if (it.first == "epoch")
		{
			epoch = std::stoi(it.second);
		}	else {
			if (it.first != "opt_type")
			{
				if (it.second == "true")
				{
					opt->SetParam(it.first, true);
				}	else if (it.second == "false")
				{
					opt->SetParam(it.first, false);
				}	else {
					opt->SetParam(it.first, it.second/*std::stof(it.second)*/);
				}
			}
		}
	}
}

template <typename DatasetType>
std::list<std::pair<std::string, std::string>> TMxDNNTrainer<DatasetType> :: ParseKeyValues(const std::string &s)
{
	size_t	pos1 = 0,
		pos2 = 0;
	std::pair<std::string, std::string> sp;
	std::list<std::pair<std::string, std::string>> result;
	do {
		pos1 = s.find("=", pos2);
		if (pos1 == std::string::npos)
			break;
		sp.first = s.substr(pos2, pos1 - pos2);
		pos2 = s.find("\n", pos1 + 1) + 1;
		sp.second = s.substr(pos1 + 1, pos2 - pos1 - 1 - 1);
		result.push_back(sp);
	} while (pos2 != std::string::npos);
	return result;
}