#include "model_executor.h"

ModelExecutor :: ModelExecutor(const ModelExecutorProperties &properties, mxnet::cpp::Symbol * net, 	mxnet::cpp::Context * context /*= nullptr*/)
{
	ExecutorProperties = properties;
	Net = net;

	if (context != nullptr)
	{
		Context = context;
	}	else { //Создать контекст
		Context = new mxnet::cpp::Context(mxnet::cpp::DeviceType::kCPU, 0);
		MXSetNumOMPThreads(std::thread::hardware_concurrency());
		int num_gpu = 0;
		MXGetGPUCount(&num_gpu);
		int batch_size = properties.BatchSize;
		#if !MXNET_USE_CPU
		if (num_gpu > 0)
		{
			delete Context;
			Context = new mxnet::cpp::Context(mxnet::cpp::DeviceType::kGPU, 0);
			batch_size = properties.BatchSize;
		}
		#endif
	}
}

///Загрузка весов сети или их начальная инициализация или копирование из другого экзекьютора
void ModelExecutor :: InitializeNet(std::vector<std::string> &arg_names, ModelExecutor * exec /*= nullptr*/)
{
	data_shape = mxnet::cpp::Shape(ExecutorProperties.BatchSize, ExecutorProperties.NetImageChannels, ExecutorProperties.NetImageWidth, ExecutorProperties.NetImageHeight);
	
	if (ExecutorProperties.ExecuteMode == DNNExecuteMode::CLASSIFIER)
		label_shape = mxnet::cpp::Shape(ExecutorProperties.BatchSize);
	if (ExecutorProperties.ExecuteMode == DNNExecuteMode::AUTOENCODER)
		label_shape = mxnet::cpp::Shape(ExecutorProperties.BatchSize, ExecutorProperties.NetImageChannels, ExecutorProperties.NetImageWidth, ExecutorProperties.NetImageHeight);
	if (ExecutorProperties.ExecuteMode == DNNExecuteMode::SEGMENTER)
		label_shape = mxnet::cpp::Shape(ExecutorProperties.BatchSize, ExecutorProperties.LabelChannels, ExecutorProperties.NetImageWidth, ExecutorProperties.NetImageHeight);
	if (ExecutorProperties.ExecuteMode == DNNExecuteMode::GENERATOR)
	{
		data_shape = mxnet::cpp::Shape(ExecutorProperties.BatchSize, ExecutorProperties.CodeVectorLength);
		label_shape = mxnet::cpp::Shape(ExecutorProperties.BatchSize, ExecutorProperties.NetImageChannels, ExecutorProperties.NetImageWidth, ExecutorProperties.NetImageHeight);
	}
	if (ExecutorProperties.ExecuteMode == DNNExecuteMode::GAN)
	{
		data_shape = mxnet::cpp::Shape(ExecutorProperties.BatchSize, ExecutorProperties.CodeVectorLength);
		label_shape = mxnet::cpp::Shape(ExecutorProperties.BatchSize);
	}

	ArgsMap["data"] = mxnet::cpp::NDArray(data_shape, *Context);
	ArgsMap["label"] = mxnet::cpp::NDArray(label_shape,*Context);

	if (exec != nullptr)
	{
		LG << "Copying net weights from another net";

		//Копирование весов из сети другого экзекьютора
		Net->InferArgsMap(*Context, &ArgsMap, ArgsMap/*known_args*/);
		auto am = exec->GetArgsMap();

		for (const auto &it : *am)
		{
			if (it.first == "data" || it.first == "label")
				continue;

			auto jt = ArgsMap.find(it.first);
			if (jt != ArgsMap.end())
			{
				jt->second = it.second.Copy(*Context);
			} else {
				std::cout<<"not found: "<<it.first<<std::endl;
			}
			mxnet::cpp::NDArray::WaitAll();
		}
		////!!!сбросить градиенты
		//for (auto &it : ArgGradStore)
		//{
		//	mxnet::cpp::NDArray::SampleUniform(0, 0, &it.second);			
		//}		
	} else {
		//Загрузка весов сети или их начальная инициализация
		if (std::filesystem::exists(ExecutorProperties.ModelFileName))
		{
			LG << "Loading net weights from " << ExecutorProperties.ModelFileName;
			LoadModelParameters(ExecutorProperties.ModelFileName, ArgsMap, ArgGradStore, GradReqType, AuxMap, *Context);
		}	else {
			Net->InferArgsMap(*Context, &ArgsMap, ArgsMap/*known_args*/);
			//Net->InferExecutorArrays(*Context, &arg_arrays, &grad_arrays, &grad_reqs, &aux_arrays, args_map, arg_grad_store, grad_req_type, aux_map);
			LG << "Initialize params with xavier";
			mxnet::cpp::Xavier xavier = mxnet::cpp::Xavier(mxnet::cpp::Xavier::gaussian, mxnet::cpp::Xavier::avg, 3.);
			for (auto &arg : ArgsMap)
			{
				if (arg.first == "data" || arg.first == "label")
					continue;
				xavier(arg.first, &arg.second);
			}
			mxnet::cpp::NDArray::WaitAll();
		}
	}

	//Exec = std::unique_ptr<mxnet::cpp::Executor>(Net->SimpleBind(*Context, ArgsMap));
	Exec = std::unique_ptr<mxnet::cpp::Executor>(Net->SimpleBind(*Context, ArgsMap, ArgGradStore, GradReqType/*std::map<std::string, mxnet::cpp::OpReqType>()*/, AuxMap));

	arg_names = Net->ListArguments();		//Нужно для последующей фильтрации, хотя это можно делать и на месте, но тогда этот параметр нужно исключить
}


void ModelExecutor :: SetArguments(const std::vector<mx_float> &batch_samples, const std::vector<mx_float> &batch_labels /*= std::vector<mx_float>()*/)
{
	Exec->arg_dict()["data"].SyncCopyFromCPU(batch_samples.data(), data_shape.Size());

	if (!batch_labels.empty())
	{
		if (ExecutorProperties.ExecuteMode == DNNExecuteMode::CLASSIFIER || 
			ExecutorProperties.ExecuteMode == DNNExecuteMode::SEGMENTER ||
			ExecutorProperties.ExecuteMode == DNNExecuteMode::GENERATOR ||
			ExecutorProperties.ExecuteMode == DNNExecuteMode::GAN
			)
			Exec->arg_dict()["label"].SyncCopyFromCPU(batch_labels.data(), label_shape.Size());

		if (ExecutorProperties.ExecuteMode == DNNExecuteMode::AUTOENCODER)
		{
			Exec->arg_dict()["label"].SyncCopyFromCPU(batch_samples.data(), label_shape.Size());
		}
	}
	mxnet::cpp::NDArray::WaitAll();
}

mxnet::cpp::NDArray ModelExecutor :: Execute(
	std::vector<mx_float> &batch_samples, 
	std::vector<mx_float> &batch_labels, /*= std::vector<mx_float>()*/
	bool is_train /*= true*/
	//const mxnet::cpp::Context * output_context /*= nullptr*/
){
	mxnet::cpp::NDArray result;
	SetArguments(batch_samples, batch_labels);
	Exec->Forward(is_train);		//!!!Пришлось оставить этот параметр true так как иначе ничего не работает возможно бага моей версии mxnet

	//if (output_context == nullptr)
	//	result = Exec->outputs[0].Copy(*Context);
	//else
	//	result = Exec->outputs[0].Copy(*output_context);

	mxnet::cpp::Context context_cpu(mxnet::cpp::DeviceType::kCPU, 0);

	result = Exec->outputs[0].Copy(context_cpu);
	mxnet::cpp::NDArray::WaitAll();

	//ShowImageFromBatch("data1", Exec->arg_dict()["data"], ExecutorProperties.NetImageWidth, ExecutorProperties.NetImageHeight, 3);
	//ShowImageFromBatch("output", Exec->outputs[0], ExecutorProperties.NetImageWidth, ExecutorProperties.NetImageHeight, 1);
	//ShowImageFromBatch("label", Exec->arg_dict()["label"], ExecutorProperties.NetImageWidth, ExecutorProperties.NetImageHeight, 1);
	//cv::waitKey(0);

	return result;
}

///The following function loads the model parameters.
void ModelExecutor :: LoadModelParameters(
	const std::string &model_parameters_file,
	std::map<std::string, mxnet::cpp::NDArray> &args_map,
	std::map< std::string, mxnet::cpp::NDArray > &arg_grad_store,
	std::map< std::string, mxnet::cpp::OpReqType> &grad_req_type, 
	std::map<std::string, mxnet::cpp::NDArray> &aux_map,
	mxnet::cpp::Context &context
) {
	if (!std::filesystem::exists(model_parameters_file))
	{
		LG << "Parameter file " << model_parameters_file << " does not exist";
		throw std::runtime_error("Model parameters does not exist");
	}
	LG << "Loading the model parameters from " << model_parameters_file << std::endl;
	std::map<std::string, mxnet::cpp::NDArray> parameters;
	mxnet::cpp::NDArray::Load(model_parameters_file, 0, &parameters);
	for (const auto &it : parameters)
	{
		if (it.first.substr(0, 4) == "arg:")
		{
			std::string name = it.first.substr(4, it.first.size() - 4);
			args_map[name] = it.second.Copy(context);
		}
		//if (it.first.substr(0, 4) == "ags:")
		//{
		//	std::string name = it.first.substr(4, it.first.size() - 4);
		//	arg_grad_store[name] = it.second.Copy(context);
		//}
		//if (it.first.substr(0, 4) == "grt:")
		//{
		//	std::string name = it.first.substr(4, it.first.size() - 4);
		//	grad_req_type[name] = it.second.Copy(context);
		//}

		if (it.first.substr(0, 4) == "aux:")
		{
			std::string name = it.first.substr(4, it.first.size() - 4);
			aux_map[name] = it.second.Copy(context);
		}
	}
	mxnet::cpp::NDArray::WaitAll();
}

std::map<std::string, mxnet::cpp::NDArray> ModelExecutor :: GetParamsMap()
{
	return GetParamsMap(ArgsMap, ArgGradStore, GradReqType, AuxMap, *Context);
}

std::map<std::string, mxnet::cpp::NDArray> ModelExecutor :: GetParamsMap (
	std::map<std::string, mxnet::cpp::NDArray> &args_map,
	std::map< std::string, mxnet::cpp::NDArray> &arg_grad_store,
	std::map< std::string, mxnet::cpp::OpReqType> &grad_req_type,
	std::map<std::string, mxnet::cpp::NDArray> &aux_map,
	mxnet::cpp::Context &context
){
	//The whole trained model is just a dictionary of array name to ndarrays.arguments starts with 'arg:' 
	//and auxiliary states starts with 'aux:'
	std::map<std::string, mxnet::cpp::NDArray> result;
	for (auto &it : args_map)
		if (it.first != "data" && it.first != "label")
		{
			result[std::string("arg:") + it.first] = it.second.Copy(context);
		}
	//for (auto &it : arg_grad_store)
	//	{
	//		parameters[std::string("ags:") + it.first] = it.second.Copy(context);
	//	}
	for (auto &it : aux_map)
	{
		result[std::string("aux:") + it.first] = it.second.Copy(context);
	}
	mxnet::cpp::NDArray::WaitAll();
	return result;
}

void ModelExecutor :: SaveModelParameters(const std::string &model_parameters_file)
{
	return SaveModelParameters(model_parameters_file, ArgsMap, ArgGradStore, GradReqType, AuxMap, *Context);
}

void ModelExecutor :: SaveModelParameters(
	const std::string &model_parameters_file,
	std::map<std::string, mxnet::cpp::NDArray> &args_map,
	std::map< std::string, mxnet::cpp::NDArray > &arg_grad_store,
	std::map< std::string, mxnet::cpp::OpReqType> &grad_req_type, 
	std::map<std::string, mxnet::cpp::NDArray> &aux_map,
	mxnet::cpp::Context &context
) {
	LG << "Saving the model parameters to " << model_parameters_file << std::endl;
	//The whole trained model is just a dictionary of array name to ndarrays.arguments starts with 'arg:' 
	//and auxiliary states starts with 'aux:'
	std::map<std::string, mxnet::cpp::NDArray> parameters = GetParamsMap(args_map, arg_grad_store, grad_req_type, aux_map, context);
	mxnet::cpp::NDArray::Save(model_parameters_file, parameters);
}