#pragma once

#include <string>
#include <map>
#include <filesystem>
#include <memory>
#include <thread>
#include <algorithm>


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

enum class TMxDNNTrainerOptimizer { ADAM, SGD };
enum class TMxDNNTrainerMode { CLASSIFIER, AUTOENCODER };

struct TMxDNNTrainerProperties {
	int DatasetImageWidth,
			DatasetImageHeight,
			NetImageWidth,
			NetImageHeight,
			MaxEpoch,
			BatchSize,
			EpochStep;						//Здесь указывается число step итераций(батчей) после которого learning rate уменьшается в factor раз
	float StartLearningRate,
				FinalLearningRate,
				WeightDecay,
				Err;
	std::string ModelFileName,
							StateFileName;
	TMxDNNTrainerOptimizer TrainerOptimizer;
	TMxDNNTrainerMode TrainerMode;
};

template <typename DatasetType>
class TMxDNNTrainer {
protected:
	TMxDNNTrainerProperties Properties;
	std::unique_ptr<mxnet::cpp::Context> _context;
	mxnet::cpp::Context * Context;
	DatasetType * Dataset;
	mxnet::cpp::Symbol * Net;
	std::unique_ptr <mxnet::cpp::Executor> exec;
	std::vector<std::string> arg_names;
	mxnet::cpp::Shape data_shape,
										label_shape;
	std::map<std::string, mxnet::cpp::NDArray> args_map;
	std::map<std::string, mxnet::cpp::NDArray> aux_map;

	///The following function loads the model parameters.
	static void LoadModelParameters(
		const std::string &model_parameters_file,
		std::map<std::string, mxnet::cpp::NDArray> &args_map,
		std::map<std::string, mxnet::cpp::NDArray> &aux_map,
		mxnet::cpp::Context &context
	);

	static void SaveModelParameters(
		const std::string &model_parameters_file,
		std::map<std::string, mxnet::cpp::NDArray> &args_map,
		std::map<std::string, mxnet::cpp::NDArray> &aux_map,
		mxnet::cpp::Context &context
	);

	std::list<std::pair<std::string, std::string>> ParseKeyValues(const std::string &s);
	void InitializeNet();
	static mxnet::cpp::NDArray ResizeInput(mxnet::cpp::NDArray data, const mxnet::cpp::Shape new_shape, const int dataset_image_width, const int dataset_image_height);
public:
	TMxDNNTrainer(
		TMxDNNTrainerProperties &properties,
		DatasetType * dataset,
		mxnet::cpp::Symbol * net,
		mxnet::cpp::Context * context = nullptr
	);

	void Train();

	void InitializeState(std::unique_ptr<mxnet::cpp::Optimizer> &opt);
	void SaveState(std::unique_ptr<mxnet::cpp::Optimizer> &opt, const int epoch);
	void LoadState(std::unique_ptr<mxnet::cpp::Optimizer> &opt, int &epoch);
};



template <typename DatasetType>
TMxDNNTrainer<DatasetType> :: TMxDNNTrainer(
	TMxDNNTrainerProperties &properties,
	DatasetType * dataset,
	mxnet::cpp::Symbol * net,
	mxnet::cpp::Context * context /*= nullptr*/
) {
	Properties = properties;
	Dataset = dataset;
	Net = net;
	if (context != nullptr)
	{
		Context = context;
	}
	else { //Создать контекст
		_context = std::unique_ptr<mxnet::cpp::Context>(new mxnet::cpp::Context(mxnet::cpp::DeviceType::kCPU, 0));
		MXSetNumOMPThreads(std::thread::hardware_concurrency());
		int num_gpu;
		MXGetGPUCount(&num_gpu);
		int batch_size = Properties.BatchSize;
#if !MXNET_USE_CPU
		if (num_gpu > 0)
		{
			_context = std::unique_ptr<mxnet::cpp::Context>(new mxnet::cpp::Context(mxnet::cpp::DeviceType::kGPU, 0));
			batch_size = Properties.BatchSize;
		}
#endif
		Context = _context.get();
	}
}

template <typename DatasetType>
void TMxDNNTrainer<DatasetType> ::InitializeNet()
{
	data_shape = mxnet::cpp::Shape(Properties.BatchSize, 1, Properties.DatasetImageWidth, Properties.DatasetImageHeight);
	if (Properties.TrainerMode == TMxDNNTrainerMode::CLASSIFIER)
		label_shape = mxnet::cpp::Shape(Properties.BatchSize);
	if (Properties.TrainerMode == TMxDNNTrainerMode::AUTOENCODER)
		label_shape = mxnet::cpp::Shape(Properties.BatchSize, 1, Properties.DatasetImageWidth, Properties.DatasetImageHeight);

	//Загрузка весов сети или их начальная инициализация
	if (std::filesystem::exists(Properties.ModelFileName))
	{
		std::cout << "Loading net weights from " << Properties.ModelFileName << std::endl;
		TMxDNNTrainer::LoadModelParameters(Properties.ModelFileName, args_map, aux_map, *Context);
		args_map["data"] = mxnet::cpp::NDArray(data_shape, *Context);
		args_map["label"] = mxnet::cpp::NDArray(label_shape, *Context);
		Net->InferArgsMap(*Context, &args_map, args_map);
	}	else {
		args_map["data"] = mxnet::cpp::NDArray(data_shape, *Context);
		args_map["label"] = mxnet::cpp::NDArray(label_shape, *Context);
		Net->InferArgsMap(*Context, &args_map, args_map);
		std::cout << "Initialize params with xavier" << std::endl;
		mxnet::cpp::Xavier xavier = mxnet::cpp::Xavier(mxnet::cpp::Xavier::gaussian, mxnet::cpp::Xavier::avg, 3.);
		for (auto& arg : args_map)
		{
			xavier(arg.first, &arg.second);
		}
	}

	exec = std::unique_ptr<mxnet::cpp::Executor>(Net->SimpleBind(*Context, args_map));
	arg_names = Net->ListArguments();
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
void TMxDNNTrainer<DatasetType> :: Train()
{
	auto logarithm = [](const double a, const double b) { return log(b) / log(a); };		//log b по основанию a

	std::unique_ptr<mxnet::cpp::Optimizer> opt;
	int start_epoch = 0;

	if (!std::filesystem::exists(Properties.StateFileName))
		InitializeState(opt);
	else
		LoadState(opt, start_epoch);

	const float factor = 0.1;
	int step = Properties.EpochStep * Dataset->Size() / Properties.BatchSize;
	//Здесь указывается число step итераций(батчей) после которого learning rate уменьшается в factor раз
	std::unique_ptr<mxnet::cpp::LRScheduler> lr_sch(new mxnet::cpp::FactorScheduler(step, factor, Properties.FinalLearningRate));
	opt->SetLRScheduler(std::move(lr_sch));
	
	for (auto it : args_map)
		std::cout << it.first << std::endl;
	std::cout << std::endl;

	// Create metrics
	mxnet::cpp::Accuracy	train_acc,
		val_acc;
	mxnet::cpp::LogLoss logloss_train,
		logloss_val;
	mxnet::cpp::MAE mae;
	mxnet::cpp::RMSE rmse;

	//ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
	std::vector<mx_float> batch_samples;
	std::vector<mx_float> batch_labels;
	mxnet::cpp::NDArray unresized_data(data_shape, *Context);

	for (int epoch = start_epoch; epoch < std::min<float>(Properties.MaxEpoch, (logarithm(1. / factor, Properties.StartLearningRate) - logarithm(1. / factor, Properties.FinalLearningRate) + 1) * Properties.EpochStep); ++epoch)
	{
		LG << "Epoch: " << epoch<< " of "
			<< std::min<float>(Properties.MaxEpoch, (logarithm(1. / factor, Properties.StartLearningRate) - logarithm(1. / factor, Properties.FinalLearningRate) + 1) * Properties.EpochStep)
			<< "; " << opt->Serialize();

		train_acc.Reset();
		int iter = 0;

		for (size_t dit = 0; dit <= Dataset->Size() / Properties.BatchSize; dit++)		//!!!Ну приблизительно эпоха
		{
			Dataset->GetRandomSampleBatch(batch_samples, batch_labels, Properties.BatchSize, *Context);
			
			unresized_data.SyncCopyFromCPU(batch_samples.data(), Properties.BatchSize /** mat.channels()*/ * Properties.NetImageWidth * Properties.NetImageWidth);
			unresized_data.CopyTo(&args_map["data"]);

			//TMxDNNTrainer::ResizeInput(unresized_data, data_shape, Properties.DatasetImageWidth, Properties.DatasetImageHeight).CopyTo(&args_map["data"]);
			//args_map["data"].SyncCopyFromCPU(batch_samples.data(), data_shape.Size());
			mxnet::cpp::NDArray::WaitAll();

			if (Properties.TrainerMode == TMxDNNTrainerMode::CLASSIFIER)
				exec->arg_dict()["label"].SyncCopyFromCPU(batch_labels.data(), label_shape.Size());
			if (Properties.TrainerMode == TMxDNNTrainerMode::AUTOENCODER)
			{
				unresized_data.CopyTo(&args_map["label"]);
				//TMxDNNTrainer::ResizeInput(unresized_data, label_shape, Properties.DatasetImageWidth, Properties.DatasetImageHeight).CopyTo(&args_map["label"]);
				mxnet::cpp::NDArray::WaitAll();
			}

			exec->Forward(true);
			exec->Backward();

			for (size_t i = 0; i < arg_names.size(); ++i)
			{
				if (arg_names[i] == "data" || arg_names[i] == "label") continue;
				opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
			}
			mxnet::cpp::NDArray::WaitAll();

			//Вывод одного изображения из батча
			if (Properties.TrainerMode == TMxDNNTrainerMode::AUTOENCODER)
			{
				mxnet::cpp::Context context_cpu(mxnet::cpp::DeviceType::kCPU, 0);
				auto nda = exec->outputs[0].Copy(context_cpu);
				mxnet::cpp::NDArray::WaitAll();
				cv::Mat mat = MXNDArrayToCVMat(nda, mxnet::cpp::Shape(1, 1, Properties.DatasetImageWidth, Properties.DatasetImageHeight));
				cv::Mat temp(mat.rows, mat.cols, CV_8U);
				cv::normalize(mat, temp, 0, 255, cv::NORM_MINMAX, CV_8U);
				cv::imshow("output", temp);
				auto nda2 = args_map["label"].Copy(context_cpu);
				mxnet::cpp::NDArray::WaitAll();
				mat = MXNDArrayToCVMat(nda2, mxnet::cpp::Shape(1, 1, Properties.DatasetImageWidth, Properties.DatasetImageHeight));
				cv::normalize(mat, temp, 0, 255, cv::NORM_MINMAX, CV_8U);
				cv::imshow("label", temp);
				cv::waitKey(1);
			}

			auto output = exec->outputs[0].Copy(*Context);
			mxnet::cpp::NDArray::WaitAll();
			if (Properties.TrainerMode == TMxDNNTrainerMode::CLASSIFIER)
			{
				train_acc.Update(args_map["label"], output);
				logloss_train.Reset();
				logloss_train.Update(args_map["label"], output);
				LG << "EPOCH: " << epoch << " ITER: " << iter << " Train Accuracy: " << train_acc.Get() << " Train Loss: " << logloss_train.Get();
			}
			if (Properties.TrainerMode == TMxDNNTrainerMode::AUTOENCODER)
			{
				mae.Reset();
				mae.Update(args_map["label"], output);
				rmse.Reset();
				rmse.Update(args_map["label"], output);
				LG << "EPOCH: " << epoch << " ITER: " << iter << " MAE: " << mae.Get() << " RMSE: " << rmse.Get();
			}
			++iter;
		}
		LG << "EPOCH: " << epoch << " Train Accuracy: " << train_acc.Get();

		//TMxDNNTrainer::SaveModelParameters(Properties.ModelFileName, args_map, aux_map, *Context);
		SaveState(opt, epoch + 1);


		//if (true)						//!!!Нужно сохранять лог обучения
		//{
		//	mxnet::cpp::Accuracy acu;
		//	acu.Reset();
		//	//!!!Тут переделать по порядку и с паддингом в конце И НА ТЕСТОВЫЙ ДАТАСЕТ
		//	for (size_t dit = 0; dit <= Dataset->Size() / Properties.BatchSize / /*!!!!*/ 15; dit++)
		//	{
		//		Dataset->GetRandomSampleBatch(batch_samples, batch_labels, Properties.BatchSize, *Context);
		//	
		//		unresized_data.SyncCopyFromCPU(batch_samples.data(), Properties.BatchSize /** mat.channels()*/ * Properties.NetImageWidth * Properties.NetImageWidth);
		//		TMxDNNTrainer::ResizeInput(unresized_data, data_shape, Properties.DatasetImageWidth, Properties.DatasetImageHeight).CopyTo(&args_map["data"]);
		//		//args_map["data"].SyncCopyFromCPU(batch_samples.data(), data_shape.Size());

		//		if (Properties.TrainerMode == TMxDNNTrainerMode::CLASSIFIER)
		//			exec->arg_dict()["label"].SyncCopyFromCPU(batch_labels.data(), label_shape.Size());
		//		if (Properties.TrainerMode == TMxDNNTrainerMode::AUTOENCODER)
		//			TMxDNNTrainer::ResizeInput(unresized_data, label_shape, Properties.DatasetImageWidth, Properties.DatasetImageHeight).CopyTo(&args_map["label"]);

		//		mxnet::cpp::NDArray::WaitAll();
		//		exec->Forward(false);
		//		auto output = exec->outputs[0].Copy(*Context);
		//		mxnet::cpp::NDArray::WaitAll();
		//		if (Properties.TrainerMode == TMxDNNTrainerMode::CLASSIFIER)
		//			acu.Update(args_map["label"], output);
		//	}
		//	LG << "Accuracy: " << acu.Get();
		//}

		if (Properties.TrainerMode == TMxDNNTrainerMode::CLASSIFIER)
			if (abs(train_acc.Get() - 1.) < Properties.Err)
				break;
	}
}

template <typename DatasetType>
void TMxDNNTrainer<DatasetType> :: InitializeState(std::unique_ptr<mxnet::cpp::Optimizer> &opt)
{
	InitializeNet();
	std::cout << "Initializing state with trainer properties" << std::endl;

	if (Properties.TrainerOptimizer == TMxDNNTrainerOptimizer::SGD)
	{
		std::cout << "optimizer: SGD" << std::endl;
		opt = std::unique_ptr<mxnet::cpp::Optimizer>(mxnet::cpp::OptimizerRegistry::Find("sgd"));
		opt->SetParam("lr", Properties.StartLearningRate)		//!!!Вот так параметры можно установить как достать смотреть ниже
			->SetParam("wd", Properties.WeightDecay)
			->SetParam("momentum", 0.99)
			->SetParam("rescale_grad", std::min(1., 1.0 / Properties.BatchSize))
			->SetParam("lazy_update", false);
	}

	if (Properties.TrainerOptimizer == TMxDNNTrainerOptimizer::ADAM)
	{
		std::cout << "optimizer: ADAM" << std::endl;
		opt = std::unique_ptr<mxnet::cpp::Optimizer>(mxnet::cpp::OptimizerRegistry::Find("adam"));
		opt->SetParam("lr", Properties.StartLearningRate)		//!!!Вот так параметры можно установить как достать смотреть ниже
			->SetParam("rescale_grad", std::min(1., 1.0 / Properties.BatchSize))
			->SetParam("beta1", 0.9)
			->SetParam("beta2", 0.999)
			->SetParam("epsilon", 1e-8)
			->SetParam("lazy_update", true)
			->SetParam("wd", Properties.WeightDecay);
	}
}

template <typename DatasetType>
void TMxDNNTrainer<DatasetType> :: SaveState(std::unique_ptr<mxnet::cpp::Optimizer> &opt, const int epoch)
{
	TMxDNNTrainer::SaveModelParameters(Properties.ModelFileName, args_map, aux_map, *Context);

	std::cout << "Saving state to " << Properties.StateFileName << std::endl;

	LG << "epoch = " << epoch << std::endl << opt->Serialize();

	WriteStringToFile(Properties.StateFileName, "epoch=" + std::to_string(epoch) + "\n" + opt->Serialize());
}

template <typename DatasetType>
void TMxDNNTrainer<DatasetType> :: LoadState(std::unique_ptr<mxnet::cpp::Optimizer> &opt, int &epoch)
{
	InitializeNet();

	std::cout << "Loading state from " << Properties.StateFileName << std::endl;

	std::string s;
	if (!ReadStringFromFile(Properties.StateFileName, s))
		std::runtime_error("TMxDNNTrainer :: LoadState error: can not read file " + Properties.StateFileName);

	LG << s;

	//Дальше их парсить и заполнять значения параметров
	auto kvl = ParseKeyValues(s);

	for (auto& it : kvl)
		if (it.first == "opt_type")
			opt = std::unique_ptr<mxnet::cpp::Optimizer>(mxnet::cpp::OptimizerRegistry::Find(it.second));

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
					opt->SetParam(it.first, std::stof(it.second));
				}
			}
		}
	}
}

///The following function loads the model parameters.
template <typename DatasetType>
void TMxDNNTrainer<DatasetType> :: LoadModelParameters(
	const std::string& model_parameters_file,
	std::map<std::string, mxnet::cpp::NDArray>& args_map,
	std::map<std::string, mxnet::cpp::NDArray>& aux_map,
	mxnet::cpp::Context& context
) {
	if (!std::filesystem::exists(model_parameters_file))
	{
		LG << "Parameter file " << model_parameters_file << " does not exist";
		throw std::runtime_error("Model parameters does not exist");
	}
	LG << "Loading the model parameters from " << model_parameters_file << std::endl;
	std::map<std::string, mxnet::cpp::NDArray> parameters;
	mxnet::cpp::NDArray::Load(model_parameters_file, 0, &parameters);
	for (const auto& it : parameters)
	{
		if (it.first.substr(0, 4) == "aux:")
		{
			std::string name = it.first.substr(4, it.first.size() - 4);
			aux_map[name] = it.second.Copy(context);
		}
		if (it.first.substr(0, 4) == "arg:")
		{
			std::string name = it.first.substr(4, it.first.size() - 4);
			args_map[name] = it.second.Copy(context);
		}
	}
	mxnet::cpp::NDArray::WaitAll();
}

template <typename DatasetType>
void TMxDNNTrainer<DatasetType> :: SaveModelParameters(
	const std::string& model_parameters_file,
	std::map<std::string, mxnet::cpp::NDArray> &args_map,
	std::map<std::string, mxnet::cpp::NDArray> &aux_map,
	mxnet::cpp::Context &context
) {
	LG << "Saving the model parameters to " << model_parameters_file << std::endl;
	//The whole trained model is just a dictionary of array name to ndarrays.arguments starts with 'arg:' 
	//and auxiliary states starts with 'aux:'
	std::map<std::string, mxnet::cpp::NDArray> parameters;

	for (auto& it : args_map)
	{
		parameters[std::string("arg:") + it.first] = it.second.Copy(context);
	}

	for (auto& it : aux_map)
	{
		parameters[std::string("aux:") + it.first] = it.second.Copy(context);
	}
	mxnet::cpp::NDArray::WaitAll();
	mxnet::cpp::NDArray::Save(model_parameters_file, parameters);
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