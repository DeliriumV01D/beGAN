#include "model_executor.h"
#include "TMxDNNTrainer.h"


///Обязательно все слои дискриминатора должны содержать в названии discriminator_
///Обязательно все слои генератора должны содержать в названии generator_
template <typename DatasetType>
class TMxGANTrainer {
protected:
	const std::string DiscrPrefix = "discriminator_",
		GenPrefix = "generator_";
	std::unique_ptr <ModelExecutor>	GeneratorExec,
		DiscriminatorExec,
		GANExec;
	std::vector<std::string>	GenArgNames,
		DiscrArgNames,
		GANArgNames;
	//Определяет обучаем сейчас генератор(1)(фиксируем дискриминатор) или обучаем дискриминатор(0)(фиксируем генератор)
	bool GanPhase;

	TMxDNNTrainerProperties Properties;

	DatasetType * TrainDataset,
	            * TestDataset;
protected:
	void InitializeMetrics(std::unique_ptr<mxnet::cpp::EvalMetric>& metric1, std::unique_ptr<mxnet::cpp::EvalMetric>& metric2);
public:
	TMxGANTrainer(
		TMxDNNTrainerProperties &properties,
		DatasetType * train_dataset,
		DatasetType * test_dataset,
		mxnet::cpp::Symbol * gan,
		mxnet::cpp::Symbol * generator,
		mxnet::cpp::Symbol * discriminator,
		mxnet::cpp::Context * context = nullptr
	);

	virtual ~TMxGANTrainer() {};

	///
	virtual void Train();

	//virtual void InitializeState(std::unique_ptr<mxnet::cpp::Optimizer> &opt);
	virtual void InitializeState(
		std::unique_ptr<mxnet::cpp::Optimizer> &discr_opt,
		std::unique_ptr<mxnet::cpp::Optimizer> &gan_opt
	);
	virtual void SaveState(mxnet::cpp::Optimizer * discr_opt, mxnet::cpp::Optimizer * gan_opt, const int epoch);
	virtual void LoadState(std::unique_ptr<mxnet::cpp::Optimizer> &discr_opt, std::unique_ptr<mxnet::cpp::Optimizer> &gan_opt, int &epoch);
};			//TMxGANTrainer

template <typename DatasetType>
TMxGANTrainer <DatasetType> :: TMxGANTrainer(
	TMxDNNTrainerProperties& properties,
	DatasetType * train_dataset,
	DatasetType * test_dataset,
	mxnet::cpp::Symbol * gan,
	mxnet::cpp::Symbol * generator,
	mxnet::cpp::Symbol * discriminator,
	mxnet::cpp::Context * context /*= nullptr*/
) {
	GanPhase = false;
	Properties = properties;

	//Здесь разобраться с размерностью входов и выходов Executor'ов
	//Все входы должны называться одинаково - data и все выходы одинаково - label

	//Generator
	ModelExecutorProperties gen_exec_properties = Properties.ExecutorProperties;
	gen_exec_properties.ExecuteMode = DNNExecuteMode::GENERATOR;		//code_vector -> image
	GeneratorExec = std::unique_ptr<ModelExecutor>(new ModelExecutor(gen_exec_properties, generator));

	//Discriminator
	ModelExecutorProperties discr_exec_properties = Properties.ExecutorProperties;
	discr_exec_properties.ExecuteMode = DNNExecuteMode::CLASSIFIER;		//image -> class[0, 1]
	DiscriminatorExec = std::unique_ptr<ModelExecutor>(new ModelExecutor(discr_exec_properties, discriminator));

	//GAN
	ModelExecutorProperties gan_exec_properties = Properties.ExecutorProperties;
	gan_exec_properties.ExecuteMode = DNNExecuteMode::GAN;					//code_vector -> class[0, 1]
	GANExec = std::unique_ptr<ModelExecutor>(new ModelExecutor(gan_exec_properties, gan));

	TrainDataset = train_dataset;
	TestDataset = test_dataset;
}

template <typename DatasetType>
void TMxGANTrainer <DatasetType> :: InitializeMetrics(std::unique_ptr<mxnet::cpp::EvalMetric> &metric1, std::unique_ptr<mxnet::cpp::EvalMetric> &metric2)
{
	metric1 = std::make_unique<mxnet::cpp::Accuracy>();
	metric2 = std::make_unique<mxnet::cpp::LogLoss>();
}

///
template <typename DatasetType>
void TMxGANTrainer <DatasetType> :: Train()
{
	auto logarithm = [](const double a, const double b) { return log(b) / log(a); };		//log b по основанию a

	std::unique_ptr<mxnet::cpp::Optimizer>	discr_opt,
		gan_opt;

	int start_epoch = 0;

	if (!std::filesystem::exists(Properties.StateFileName + "_gan") || !std::filesystem::exists(Properties.StateFileName + "_discr"))
		InitializeState(discr_opt, gan_opt);
	else
		LoadState(discr_opt, gan_opt, start_epoch);

	const float factor = 0.1f;
	int step = Properties.EpochStep * TrainDataset->Size() / Properties.ExecutorProperties.BatchSize;
	//Здесь указывается число step итераций(батчей) после которого learning rate уменьшается в factor раз
	std::unique_ptr<TMxDNNScheduler> discr_lr_sch(new TMxDNNScheduler(Properties.StartLearningRate, step, factor, Properties.FinalLearningRate, start_epoch * TrainDataset->Size() / Properties.ExecutorProperties.BatchSize));
	discr_opt->SetLRScheduler(std::move(discr_lr_sch));
	std::unique_ptr<TMxDNNScheduler> gan_lr_sch(new TMxDNNScheduler(Properties.StartLearningRate, step, factor, Properties.FinalLearningRate, start_epoch * TrainDataset->Size() / Properties.ExecutorProperties.BatchSize));
	gan_opt->SetLRScheduler(std::move(gan_lr_sch));

	//for (auto it : GANExec->GetExecutor()->arg_dict())
	//	std::cout << it.first << std::endl;
	//std::cout << std::endl;

	// Create metrics
	std::unique_ptr<mxnet::cpp::EvalMetric> metric1,
		                                      metric2;
	InitializeMetrics(metric1, metric2);

	//ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
	std::vector<mx_float> batch_samples(Properties.ExecutorProperties.BatchSize * Properties.ExecutorProperties.NetImageChannels * Properties.ExecutorProperties.NetImageWidth * Properties.ExecutorProperties.NetImageHeight);
	std::vector<mx_float> batch_labels(Properties.ExecutorProperties.BatchSize);
	mxnet::cpp::NDArray output;
	for (int epoch = start_epoch; epoch < std::min<float>((float)Properties.MaxEpoch, (float)(logarithm(1. / factor, Properties.StartLearningRate) - logarithm(1. / factor, Properties.FinalLearningRate) + 1) * Properties.EpochStep); ++epoch)
	{
		LG << "Epoch: " << epoch << " of "
			<< std::min<float>((float)Properties.MaxEpoch, (float)(logarithm(1. / factor, Properties.StartLearningRate) - logarithm(1. / factor, Properties.FinalLearningRate) + 1) * Properties.EpochStep)
			<< "; " << discr_opt->Serialize() << std::endl << gan_opt->Serialize() << std::endl;

		int iter = 0;
		//Зафиксировать веса генератора. Поочередно подавать на вход дискриминатора выход генератора и изображение из датасета. Обучать до тех пор пока различение не улучшится.
		//Зафиксировать веса дискриминатора. Подавать на вход дискриминатора только выход генератора. Обучать до тех пор пока генератор не станет чаще успешно обманывать дискриминатор.
		bool mode; //Определяет берем изображения батча из датасета(true) или из генератора(false)
		std::vector<mx_float> gan_rand(Properties.ExecutorProperties.CodeVectorLength * Properties.ExecutorProperties.BatchSize);

		if (epoch == 0)
		{
			if (epoch % 2 == 0)
				GanPhase = false;
			else
				GanPhase = true;
		}	else {
			if (GanPhase == true)
			{
				GanPhase = false;
				metric1->Reset();
				metric2->Reset();
				std::cout << "GanPhase = false. Trainig discriminator." << std::endl;
			} else {
				GanPhase = true;
				metric1->Reset();
				metric2->Reset();
				std::cout << "GanPhase = true. Trainig generator." << std::endl;
			}		
		}

		//arXiv:1406.2661v1
		//Optimizing D to completion in the inner loop of training is computationally prohibitive,
		//and on finite datasets would result in overfitting.Instead, we alternate between k steps of optimizing
		//Dand one step of optimizing G.This results in D being maintained near its optimal solution, so
		//long as G changes slowly enough.This strategy is analogous to the way that SML / PCD[31, 29]
		//training maintains samples from a Markov chain from one learning step to the next in order to avoid
		//burning in a Markov chain as part of the inner loop of learning
		for (size_t dit = 0; dit <= (GanPhase == true) * 25 + (GanPhase == false) * 50/*(GanPhase == true) * TrainDataset->Size() / Properties.ExecutorProperties.BatchSize + (GanPhase == false) * TrainDataset->Size() / Properties.ExecutorProperties.BatchSize*/; dit++)		//!!!Ну приблизительно эпоха
		{
			if (GanPhase == false)
				mode = (dit % 2) == 1;
			else
				mode = false;

			if (GanPhase == false)			//обучаем дискриминатор
			{
				//В зависимости от mode заполняем data батча из датасета(true) или из генератора(false)
				if (mode == true)
				{
					TrainDataset->GetRandomSampleBatch(batch_samples, batch_labels, Properties.ExecutorProperties.BatchSize);
					//!!!DiscriminatorExec->SetArguments(samples, labels); см ниже
				}	else {
					for (int l = 0; l < gan_rand.size(); l++)
						gan_rand[l] = static_cast<mx_float>(RandomDouble() /** 2 - 1*/);		///Привести к диапазону [-1, 1]
					mxnet::cpp::NDArray generated = GeneratorExec->Execute(gan_rand, std::vector<mx_float>(), true/*false*/);

					ShowImageFromBatch("gen", generated, Properties.ExecutorProperties.NetImageWidth, Properties.ExecutorProperties.NetImageHeight, 1, false);
					cv::waitKey(1);

					//generated(уже на cpu) -> batch_samples
					memcpy(&batch_samples[0], generated.GetData(), generated.Size() * sizeof(mx_float));
				}

				//Вместо меток класса даем признак настоящее изображение или поддельное
				for (int l = 0; l < batch_labels.size(); l++)
				{
					batch_labels[l] = mode ? 1 : 0;
					//if (mode == false)
					//	batch_labels[l] = 0.f;
					//else
					//	batch_labels[l] = 0.9f + RandomDouble() * 0.1; //label smoothing to make the discriminator training more robust 
				}

				//output = DiscriminatorExec->Execute(batch_samples, batch_labels, true); - нет Backward!
				DiscriminatorExec->SetArguments(batch_samples, batch_labels);
				DiscriminatorExec->GetExecutor()->Forward(true);
				DiscriminatorExec->GetExecutor()->Backward();

				ShowImageFromBatch("in", DiscriminatorExec->GetExecutor()->arg_dict()["data"], Properties.ExecutorProperties.NetImageWidth, Properties.ExecutorProperties.NetImageHeight, 1, false);
				cv::waitKey(1);

				for (size_t i = 0; i < DiscrArgNames.size(); ++i)
				{
					if (DiscrArgNames[i] == "data" || DiscrArgNames[i] == "label")
						continue;
					//if (non_updatable_arguments.find(DiscrArgNames[i]) == non_updatable_arguments.end())
					discr_opt->Update(static_cast<int>(i), DiscriminatorExec->GetExecutor()->arg_dict()[DiscrArgNames[i]], DiscriminatorExec->GetExecutor()->grad_dict()[DiscrArgNames[i]]);
				}

				mxnet::cpp::Context context_cpu(mxnet::cpp::DeviceType::kCPU, 0);
				output = DiscriminatorExec->GetExecutor()->outputs[0].Copy(context_cpu);
				
				DiscriminatorExec->SetArguments(batch_samples, batch_labels);
				
				mxnet::cpp::NDArray::WaitAll();

				////!!!
				//for (int l = 0; l < batch_labels.size(); l++)
				//	std::cout << output.GetData()[l] << " " << batch_labels[l] << "        ";

				//metric1->Reset();
				metric1->Update(DiscriminatorExec->GetExecutor()->arg_dict()["label"], output);
				if (metric2 != nullptr)
				{
					//metric2->Reset();
					metric2->Update(DiscriminatorExec->GetExecutor()->arg_dict()["label"], output);
				}
			}		else {						//обучаем генератор	/////////////////////////////////////////////////////////////////////////////

				for (int l = 0; l < gan_rand.size(); l++)
					gan_rand[l] = static_cast<mx_float>(RandomDouble() /** 2 - 1*/);		///Привести к диапазону [-1, 1]

				//Вместо меток класса даем признак настоящее изображение или поддельное
				//(gan_mode == 0)		//используем только выход генератора
				for (int l = 0; l < batch_labels.size(); l++)
					batch_labels[l] = mode ? 1 : 0;

				//output = GANExec->Execute(gan_rand, batch_labels, true); - нет backward
				GANExec->SetArguments(gan_rand, batch_labels);
				GANExec->GetExecutor()->Forward(true);
				GANExec->GetExecutor()->Backward();

				std::unordered_set<std::string> non_updatable_arguments = GetFilteredArgumentSet(GANArgNames, "discriminator_");

				for (size_t i = 0; i < GANArgNames.size(); ++i)
				{
					if (GANArgNames[i] == "data" || GANArgNames[i] == "label")
						continue;
					if (non_updatable_arguments.find(GANArgNames[i]) == non_updatable_arguments.end())
					{
						gan_opt->Update(static_cast<int>(i), GANExec->GetExecutor()->arg_dict()[GANArgNames[i]], GANExec->GetExecutor()->grad_dict()[GANArgNames[i]]);
					}
				}
		
				mxnet::cpp::Context context_cpu(mxnet::cpp::DeviceType::kCPU, 0);
				output = GANExec->GetExecutor()->outputs[0].Copy(context_cpu);

				GANExec->SetArguments(gan_rand, batch_labels);

				mxnet::cpp::NDArray::WaitAll();

				//metric1->Reset();
				metric1->Update(GANExec->GetExecutor()->arg_dict()["label"], output);
				if (metric2 != nullptr)
				{
					//metric2->Reset();
					metric2->Update(GANExec->GetExecutor()->arg_dict()["label"], output);
				}
			}  //(GanPhase == true)			//обучаем дискриминатор/генератор

			LG << "EPOCH: " << epoch << " ITER: " << iter << " Train Accuracy: " << metric1->Get() << " Train Loss: " << metric2->Get() << "; gen " << mode;

			++iter;
		}

		if (GanPhase == false)			//обучаем дискриминатор
		{
			//передать новые веса дискриминатора в GAN
			GANExec->InitializeNet(GANArgNames, DiscriminatorExec.get());
		} else {										// обучаем генератор
			//передать новые веса из GAN в Generator
			GeneratorExec->InitializeNet(GenArgNames, GANExec.get());

			//Сохранить состояние
			SaveState(discr_opt.get(), gan_opt.get(), epoch + 1);
		}

		///!!!Вывод статистики по тестовому датасету и сохранение лога обучения

		///!!!Достижение заданной точности
		//if (TerminationConditions(metric1))
		//	break;
	}
}

template <typename DatasetType>
void TMxGANTrainer <DatasetType> :: InitializeState(
	std::unique_ptr<mxnet::cpp::Optimizer> &discr_opt,
	std::unique_ptr<mxnet::cpp::Optimizer> &gan_opt
) {
	//
	GANExec->InitializeNet(GANArgNames);

	////!!!Переделать этот костыль!!! Если загружать только веса генератора, то тоже в gan а оттуда скопируется в gen
	//std::string temp = GANExec->ExecutorProperties.ModelFileName;
	//GANExec->ExecutorProperties.ModelFileName = "resnet.dat";
	//GANExec->InitializeNet(GenArgNames);
	//GANExec->ExecutorProperties.ModelFileName = temp;
	
	//Копирование весов данного списка слоев из одной сетки в другую
	GeneratorExec->InitializeNet(GenArgNames, GANExec.get());
	DiscriminatorExec->InitializeNet(DiscrArgNames, GANExec.get());

	LG << "Initializing state with trainer properties";

	if (Properties.TrainerOptimizer == TMxDNNTrainerOptimizer::SGD)
	{
		LG << "optimizer: SGD";
		discr_opt.reset(mxnet::cpp::OptimizerRegistry::Find("sgd"));
		discr_opt->SetParam("lr", Properties.StartLearningRate)		//!!!Вот так параметры можно установить как достать смотреть ниже
			->SetParam("wd", Properties.WeightDecay)
			->SetParam("momentum", 0.99)
			->SetParam("rescale_grad", std::min(1., 1.0 / Properties.ExecutorProperties.BatchSize))
			->SetParam("lazy_update", false);
		gan_opt.reset(mxnet::cpp::OptimizerRegistry::Find("sgd"));
		gan_opt->SetParam("lr", Properties.StartLearningRate)		//!!!Вот так параметры можно установить как достать смотреть ниже
			->SetParam("wd", Properties.WeightDecay)
			->SetParam("momentum", 0.99)
			->SetParam("rescale_grad", std::min(1., 1.0 / Properties.ExecutorProperties.BatchSize))
			->SetParam("lazy_update", false);
	}

	if (Properties.TrainerOptimizer == TMxDNNTrainerOptimizer::ADAM)
	{
		LG << "optimizer: ADAM";
		discr_opt.reset(mxnet::cpp::OptimizerRegistry::Find("adam"));
		discr_opt->SetParam("lr", Properties.StartLearningRate)		//!!!Вот так параметры можно установить как достать смотреть ниже
			->SetParam("rescale_grad", std::min(1., 1.0 / Properties.ExecutorProperties.BatchSize))
			->SetParam("beta1", 0.9)
			->SetParam("beta2", 0.999)
			->SetParam("epsilon", 1e-8)
			->SetParam("lazy_update", false)
			->SetParam("wd", Properties.WeightDecay);
		gan_opt.reset(mxnet::cpp::OptimizerRegistry::Find("adam"));
		gan_opt->SetParam("lr", Properties.StartLearningRate)		//!!!Вот так параметры можно установить как достать смотреть ниже
			->SetParam("rescale_grad", std::min(1., 1.0 / Properties.ExecutorProperties.BatchSize))
			->SetParam("beta1", 0.9)
			->SetParam("beta2", 0.999)
			->SetParam("epsilon", 1e-8)
			->SetParam("lazy_update", false)
			->SetParam("wd", Properties.WeightDecay);
	}
}

template <typename DatasetType>
void TMxGANTrainer<DatasetType> :: SaveState(
	mxnet::cpp::Optimizer * discr_opt,
	mxnet::cpp::Optimizer * gan_opt,
	const int epoch
) {
	GANExec->SaveModelParameters(Properties.ExecutorProperties.ModelFileName);
	LG << "Saving GAN state to " << Properties.StateFileName;

	LG << "epoch = " << epoch << std::endl << discr_opt->Serialize() << std::endl << gan_opt->Serialize();

	WriteStringToFile(Properties.StateFileName + "_discr", "epoch=" + std::to_string(epoch) + "\n" + discr_opt->Serialize());
	WriteStringToFile(Properties.StateFileName + "_gan", "epoch=" + std::to_string(epoch) + "\n" + gan_opt->Serialize());
}

template <typename DatasetType>
void TMxGANTrainer<DatasetType> :: LoadState(
	std::unique_ptr<mxnet::cpp::Optimizer>& discr_opt,
	std::unique_ptr<mxnet::cpp::Optimizer>& gan_opt,
	int& epoch)
{
	GANExec->InitializeNet(GANArgNames);
	LG << "Loading GAN state from " << Properties.StateFileName;

	//Копирование весов данного списка слоев из одной сетки в другую
	GeneratorExec->InitializeNet(GenArgNames, GANExec.get());
	DiscriminatorExec->InitializeNet(DiscrArgNames, GANExec.get());

	std::string s;
	if (!ReadStringFromFile(Properties.StateFileName + "_discr", s))
		std::runtime_error("TMxGANTrainer :: LoadState error: can not read file " + Properties.StateFileName + "_discr");

	LG << s;

	//Дальше их парсить и заполнять значения параметров
	auto kvl = TMxDNNTrainer<DatasetType> ::ParseKeyValues(s);

	for (auto& it : kvl)
		if (it.first == "opt_type")
			discr_opt.reset(mxnet::cpp::OptimizerRegistry::Find(it.second));
	for (auto& it : kvl)
	{
		if (it.first == "epoch")
		{
			epoch = std::stoi(it.second);
		}
		else {
			if (it.first != "opt_type")
			{
				if (it.second == "true")
				{
					discr_opt->SetParam(it.first, true);
				}
				else if (it.second == "false")
				{
					discr_opt->SetParam(it.first, false);
				}
				else {
					discr_opt->SetParam(it.first, it.second/*std::stof(it.second)*/);
				}
			}
		}
	}

	if (!ReadStringFromFile(Properties.StateFileName + "_gan", s))
		std::runtime_error("TMxGANTrainer :: LoadState error: can not read file " + Properties.StateFileName + "_gan");

	LG << s;

	//Дальше их парсить и заполнять значения параметров
	kvl = TMxDNNTrainer<DatasetType> ::ParseKeyValues(s);

	for (auto& it : kvl)
		if (it.first == "opt_type")
			gan_opt.reset(mxnet::cpp::OptimizerRegistry::Find(it.second));
	for (auto& it : kvl)
	{
		if (it.first == "epoch")
		{
			epoch = std::stoi(it.second);
		}
		else {
			if (it.first != "opt_type")
			{
				if (it.second == "true")
				{
					gan_opt->SetParam(it.first, true);
				}
				else if (it.second == "false")
				{
					gan_opt->SetParam(it.first, false);
				}
				else {
					gan_opt->SetParam(it.first, it.second/*std::stof(it.second)*/);
				}
			}
		}
	}
}