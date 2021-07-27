#pragma once

#include "CommonDefinitions.h"

DISABLE_WARNING_PUSH

#if defined(_MSC_VER)
DISABLE_WARNING(4267)
DISABLE_WARNING(4305)
DISABLE_WARNING(4251)
DISABLE_WARNING(4244)
#endif

#include "mxnet-cpp/MxNetCpp.h"
#include "mxnet-cpp/metric.h"

DISABLE_WARNING_POP

#include <filesystem>

enum class DNNExecuteMode { CLASSIFIER, AUTOENCODER, SEGMENTER, GENERATOR, GAN };

///
struct ModelExecutorProperties {
	int NetImageChannels,
			NetImageWidth,
			NetImageHeight,
			BatchSize,
			LabelChannels,
			CodeVectorLength;
	DNNExecuteMode ExecuteMode;
	std::string ModelFileName;
};

///
class ModelExecutor {
protected:
	mxnet::cpp::Symbol * Net;
	std::unique_ptr <mxnet::cpp::Executor> Exec;
	mxnet::cpp::Context * Context;

	mxnet::cpp::Shape data_shape,
										label_shape;
	std::vector<mxnet::cpp::NDArray> ArgArrays;
	std::vector<mxnet::cpp::NDArray> GradArrays; 
	std::vector<mxnet::cpp::OpReqType> GradReqs;
	std::vector<mxnet::cpp::NDArray> AuxArrays;
	std::map<std::string, mxnet::cpp::NDArray> ArgsMap;
	std::map< std::string, mxnet::cpp::NDArray> ArgGradStore;
	std::map< std::string, mxnet::cpp::OpReqType> GradReqType;
	std::map<std::string, mxnet::cpp::NDArray> AuxMap;

public:
	//!!!Вернуть в protected
	ModelExecutorProperties ExecutorProperties;

	ModelExecutor(const ModelExecutorProperties &properties, mxnet::cpp::Symbol * net, 	mxnet::cpp::Context * context = nullptr);
	virtual ~ModelExecutor(){};
	///Загрузка весов сети или их начальная инициализация
	virtual void InitializeNet(std::vector<std::string> &arg_names, ModelExecutor * exec = nullptr);
	
	virtual void SetArguments(const std::vector<mx_float> &batch_samples, const std::vector<mx_float> &batch_labels = std::vector<mx_float>());
	//virtual void Train(const std::vector<mx_float> &batch_samples, const std::vector<mx_float> &batch_labels);
	virtual mxnet::cpp::NDArray Execute(
		std::vector<mx_float> &batch_samples, 
		std::vector<mx_float> &batch_labels = std::vector<mx_float>(),
		bool is_train = true
		//const mxnet::cpp::Context * output_context
	);

	///The following function loads the model parameters.
	virtual void LoadModelParameters(
		const std::string &model_parameters_file,
		std::map <std::string, mxnet::cpp::NDArray> &args_map,
		std::map <std::string, mxnet::cpp::NDArray > &arg_grad_store,
		std::map <std::string, mxnet::cpp::OpReqType> &grad_req_type, 
		std::map <std::string, mxnet::cpp::NDArray> &aux_map,
		mxnet::cpp::Context &context
	);

	virtual std::map<std::string, mxnet::cpp::NDArray> GetParamsMap(
		std::map <std::string, mxnet::cpp::NDArray> &args_map,
		std::map <std::string, mxnet::cpp::NDArray> &arg_grad_store,
		std::map <std::string, mxnet::cpp::OpReqType> &grad_req_type,
		std::map <std::string, mxnet::cpp::NDArray> &aux_map,
		mxnet::cpp::Context& context
	);

	virtual std::map<std::string, mxnet::cpp::NDArray> GetParamsMap();

	virtual void SaveModelParameters(const std::string& model_parameters_file);

	virtual void SaveModelParameters(
		const std::string &model_parameters_file,
		std::map <std::string, mxnet::cpp::NDArray> &args_map,
		std::map <std::string, mxnet::cpp::NDArray > &arg_grad_store,
		std::map <std::string, mxnet::cpp::OpReqType> &grad_req_type, 
		std::map <std::string, mxnet::cpp::NDArray> &aux_map,
		mxnet::cpp::Context &context
	);

	mxnet::cpp::Context * GetContext()
	{
		return Context;
	}

	void SetArg(const std::string &arg_name, const mxnet::cpp::NDArray * arg_value)
	{
		ArgsMap[arg_name] = arg_value->Copy(*Context);
	}

	void SetAux(const std::string &aux_name, const mxnet::cpp::NDArray * aux_value)
	{
		AuxMap[aux_name] = aux_value->Copy(*Context);
	}

	mxnet::cpp::Executor * GetExecutor()
	{
		return Exec.get();
	}

	std::map<std::string, mxnet::cpp::NDArray> * GetArgsMap()
	{
		return &ArgsMap;
	}
};