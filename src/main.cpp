//#define DO_NOT_USE_DLIB

#include <chrono>
#include <string>
#include <fstream>
#include <vector>
#include <filesystem>
#include <unordered_set>

#include "TRandomDouble.h"
#include "TMnistDataset.h"
#include "TDataset.h"
#include "TFacesDataset.h"
#include "TMxDNNTrainer.h"
#include "TMxGANTrainer.h"
#include "CommonDefinitions.h"


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


const int DATASET_IMAGE_SIZE = 96;
const int NET_IMAGE_SIZE = 96;
const int MAX_EPOCH = 100000;
const float START_LEARNING_RATE = 1e-4f;
const float WEIGHT_DECAY = 1e-5f;	//1e-6;//0.;
const float ERR = 1.e-5f;
const int BATCH_SIZE = 4;

static const std::string DATA_DIR = "D:\\Delirium\\PROJECTS\\CheckPoint Smart\\FaceRecognizer\\FaceRecognizer\\Data";
static const std::string TEST_DATA_DIR = "D:\\Delirium\\PROJECTS\\CheckPoint Smart\\FaceRecognizer\\FaceRecognizer\\Test";
static const bool USE_DATA_AUGMENTATION = true;
static const std::string POS_MODEL_FILENAME = "D:\\Delirium\\PROJECTS\\CheckPoint Smart\\FaceRecognizer\\FaceRecognizer\\FaceRecognizer\\shape_predictor_68_face_landmarks.dat";
static const std::string MODEL_FILENAME = "gan.dat";
static const std::string STATE_FILENAME = "gan_sync";
static const int FR_DESCRIPTOR_SIZE = 160;

mxnet::cpp::Symbol Decoder(
	const std::string &name,
	mxnet::cpp::Symbol input,
	int code_vector_length,
	int num_filter = 16,
	mx_float bn_momentum = 0.9
){
	auto input_shape = mxnet::cpp::Shape(num_filter * 64, 1, 1);

	auto fc1 = mxnet::cpp::Operator("FullyConnected")
		.SetParam("num_hidden", input_shape.Size())					//размер внутри батча
		.SetParam("no_bias", true)
		.SetInput("data", input)
		.CreateSymbol(name + "_fc1");
	auto relu1 = mxnet::cpp::Operator("Activation")
		.SetParam("act_type", "relu")
		.SetInput("data", fc1)
		.CreateSymbol(name + "_relu1");

	//Размер всех батчей
	auto reshaped = mxnet::cpp::Reshape(name + "_reshape", relu1, mxnet::cpp::Shape(BATCH_SIZE, num_filter * 64, 1, 1));  

	mxnet::cpp::Symbol conv2 = GetTrConv(name + "_conv2", reshaped, num_filter * 32, mxnet::cpp::Shape(3, 3), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(0, 0), true, bn_momentum);  //столбцы [nbatches, 1024, 1, 1] -> [nbatches, 512, 3, 3]

	mxnet::cpp::Symbol block12 = MakeTrBlock(name + "_block12", conv2, num_filter * 32, false, bn_momentum);																						//[nbatches, 512, 3, 3]
	mxnet::cpp::Symbol block11 = MakeTrBlock(name + "_block11", block12, num_filter * 32, true, bn_momentum);
	
	mxnet::cpp::Symbol block10 = MakeTrBlock(name + "_block10", block11, num_filter * 16, false, bn_momentum);																					//[nbatches, 256, 6, 6]
	mxnet::cpp::Symbol block9 = MakeTrBlock(name + "_block9", block10, num_filter * 16, true, bn_momentum);

	mxnet::cpp::Symbol block8 = MakeTrBlock(name + "_block8", block9, num_filter * 8, false, bn_momentum);
	mxnet::cpp::Symbol block7 = MakeTrBlock(name + "_block7", block8, num_filter * 8, true, bn_momentum);

	mxnet::cpp::Symbol block5 = MakeTrBlock(name + "_block5", block7, num_filter * 4, false, bn_momentum);
	mxnet::cpp::Symbol block4 = MakeTrBlock(name + "_block4", block5, num_filter * 4, true, bn_momentum);

	mxnet::cpp::Symbol block2 = MakeTrBlock(name + "_block2", block4, num_filter * 2, false, bn_momentum);
	mxnet::cpp::Symbol block1 = MakeTrBlock(name + "_block1", block2, num_filter * 2, true, bn_momentum);
	//mxnet::cpp::Symbol conv1 = GetTrConv(name + "_conv1", block1, num_filter, mxnet::cpp::Shape(5, 5), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(2, 2), true, bn_momentum);
	mxnet::cpp::Symbol conv0 = GetTrConv(name + "_conv0", block1, num_filter, mxnet::cpp::Shape(5, 5), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(2, 2), true, bn_momentum);
	
	mxnet::cpp::Symbol conv1x1 = GetTrConv(name + "_conv1x1", conv0, 1, mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(0, 0), true, bn_momentum);										//[1 96 96]		// 1x1 convolution
	return conv1x1;
}

mxnet::cpp::Symbol Encoder(
	const std::string &name,
	mxnet::cpp::Symbol input,
	int code_vector_length,
	int num_filter = 16,
	mx_float bn_momentum = 0.9
){
	mxnet::cpp::Symbol conv0 = GetConv(name + "_conv0", input, num_filter, mxnet::cpp::Shape(5, 5), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(2, 2), true, bn_momentum);
	//mxnet::cpp::Symbol conv1 = GetConv(name + "_conv1", conv0, num_filter, mxnet::cpp::Shape(5, 5), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(2, 2), true, bn_momentum);
	//mxnet::cpp::Symbol block1 = MakeBlock(name + "_block1", conv0, num_filter, true, bn_momentum);
	mxnet::cpp::Symbol block2 = MakeBlock(name + "_block2", conv0, num_filter * 2, false, bn_momentum);

	//mxnet::cpp::Symbol block4 = MakeBlock(name + "_block4", block2, num_filter * 2, true, bn_momentum);
	mxnet::cpp::Symbol block5 = MakeBlock(name + "_block5", block2, num_filter * 4, false, bn_momentum);

	//mxnet::cpp::Symbol block7 = MakeBlock(name + "_block7", block5, num_filter * 4, true, bn_momentum);
	mxnet::cpp::Symbol block8 = MakeBlock(name + "_block8", block5, num_filter * 8, false, bn_momentum);

	//mxnet::cpp::Symbol block9 = MakeBlock(name + "_block9", block8, num_filter * 8, true, bn_momentum);
	mxnet::cpp::Symbol block10 = MakeBlock(name + "_block10", block8, num_filter * 16, false, bn_momentum);		//[nbatches, 256, 6, 6]
	
	//mxnet::cpp::Symbol block11 = MakeBlock(name + "_block11", block10, num_filter * 16, true, bn_momentum);
	mxnet::cpp::Symbol block12 = MakeBlock(name + "_block12", block10, num_filter * 32, false, bn_momentum);		//[nbatches, 512, 3, 3]

	mxnet::cpp::Symbol conv2 = GetConv(name + "_conv2", block12, num_filter * 64, mxnet::cpp::Shape(3, 3), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(0, 0), true, bn_momentum);  //столбцы [nbatches, 1024, 1, 1]

	auto fc2 = mxnet::cpp::Operator("FullyConnected")
		.SetParam("num_hidden", code_vector_length)
		.SetParam("no_bias", true)
		.SetInput("data", conv2)
		.CreateSymbol(name + "_fc2");

	auto relu = mxnet::cpp::Operator("Activation")
		.SetParam("act_type", "relu")
		.SetInput("data", fc2)
		.CreateSymbol(name + "_tanh");

	return relu;																																	//[nb, code_vector_length]
	////mxnet::cpp::Symbol fused = shortcut + conv2;
	////return Activation(name + "_relu", fused, "relu");
}


mxnet::cpp::Symbol MyResNetAutoEncoder(
	const std::string &name,
	int code_vector_length,
	int num_filter = 16,
	mx_float bn_momentum = 0.9
) {
	// data and label
	mxnet::cpp::Symbol data = mxnet::cpp::Symbol::Variable("data");
	mxnet::cpp::Symbol data_label = mxnet::cpp::Symbol::Variable("label");

	mxnet::cpp::Symbol gamma(name + "_gamma");
	mxnet::cpp::Symbol beta(name + "_beta");
	mxnet::cpp::Symbol mmean(name + "_mmean");
	mxnet::cpp::Symbol mvar(name + "_mvar");

	mxnet::cpp::Symbol zscore = BatchNorm(name + "_zscore", data, gamma, beta, mmean, mvar, 0.001, bn_momentum);

	mxnet::cpp::Symbol encoder = Encoder(name + "_discriminator", zscore, code_vector_length, num_filter, bn_momentum);
	mxnet::cpp::Symbol decoder = Decoder(name + "_generator", encoder, code_vector_length, num_filter, bn_momentum);

	//return mxnet::cpp::LinearRegressionOutput(name + "_lro", decoder, data_label);
	return mxnet::cpp::MAERegressionOutput(name + "_mae_ro", decoder, data_label);
}


mxnet::cpp::Symbol MyResNetGenerator(
	const std::string &name,
	int code_vector_length,
	int num_filter = 16,
	mx_float bn_momentum = 0.9
){
	mxnet::cpp::Symbol /*rand*/ data = mxnet::cpp::Symbol::Variable("data"/*"rand"*/);
	mxnet::cpp::Symbol data_label = mxnet::cpp::Symbol::Variable("label");

	//mxnet::cpp::Symbol gamma(name + "_generator_gamma");
	//mxnet::cpp::Symbol beta(name + "_generator_beta");
	//mxnet::cpp::Symbol mmean(name + "_generator_mmean");
	//mxnet::cpp::Symbol mvar(name + "_generator_mvar");

	//mxnet::cpp::Symbol zscore = BatchNorm(name + "_generator_zscore", data, gamma, beta, mmean, mvar, 0.001, bn_momentum);

	mxnet::cpp::Symbol decoder = Decoder(name + "_generator", data/*zscore*//*rand*/, code_vector_length, num_filter, bn_momentum);
	return mxnet::cpp::MAERegressionOutput(name + "_generator_mae_ro", decoder, data_label);
}

mxnet::cpp::Symbol MyResNetDiscriminator(
	const std::string &name,
	int code_vector_length,
	int num_filter = 16,
	mx_float bn_momentum = 0.9
){
	//mxnet::cpp::Symbol phase = mxnet::cpp::Symbol::Variable("phase");	//Определяет обучаем сейчас генератор(1)(фиксируем дискриминатор) или обучаем дискриминатор(0)(фиксируем генератор)
	mxnet::cpp::Symbol data = mxnet::cpp::Symbol::Variable("data");
	mxnet::cpp::Symbol data_label = mxnet::cpp::Symbol::Variable("label");

	//mxnet::cpp::Symbol gamma(name + "_discriminator_gamma");
	//mxnet::cpp::Symbol beta(name + "_discriminator_beta");
	//mxnet::cpp::Symbol mmean(name + "_discriminator_mmean");
	//mxnet::cpp::Symbol mvar(name + "_discriminator_mvar");

	//mxnet::cpp::Symbol zscore = BatchNorm(name + "_discriminator_zscore", data, gamma, beta, mmean, mvar, 0.001, bn_momentum);

	mxnet::cpp::Symbol discriminator = Encoder(name + "_discriminator", data/*zscore*/, code_vector_length, num_filter, bn_momentum);

	//mxnet::cpp::Symbol flat = mxnet::cpp::Flatten(name + "_discriminator_flatten", discriminator);

	auto fc = mxnet::cpp::Operator("FullyConnected")
		.SetParam("num_hidden", 2)
		.SetParam("no_bias", true)
		.SetInput("data", discriminator)
		.CreateSymbol(name + "_discriminator_fc_h");

	auto relu = mxnet::cpp::Operator("Activation")
		.SetParam("act_type", "relu")
		.SetInput("data", fc)
		.CreateSymbol(name + "_discriminator_relu_h");

	return SoftmaxOutput(name + "_discriminator_softmax1", /*fc*(-1) + 1*/relu, data_label/*, 1.F, -1.F, false, false, false, mxnet::cpp::SoftmaxOutputNormalization::kBatch*/);

	//auto reshaped = mxnet::cpp::Reshape(name + "_discriminator_pool_reshape", relu, mxnet::cpp::Shape(BATCH_SIZE, 2, 1, 1));
	//mxnet::cpp::Symbol sm = Pooling(name + "_discriminator_pool", reshaped, mxnet::cpp::Shape(1, 2, 1, 1), mxnet::cpp::PoolingPoolType::kMax,
	//	false, false, mxnet::cpp::PoolingPoolingConvention::kValid/*, mxnet::cpp::Shape(1, 2, 1, 1)*/);
	//auto reshaped2 = mxnet::cpp::Reshape(name + "_discriminator_pool_reshape2", sm, mxnet::cpp::Shape(BATCH_SIZE));
	
	//auto fc2 = mxnet::cpp::Operator("FullyConnected")
	//	.SetParam("num_hidden", 1)
	//	.SetParam("no_bias", true)
	//	.SetInput("data", relu)
	//	.CreateSymbol(name + "_discriminator_fc_h2");

	//auto relu2 = mxnet::cpp::Operator("Activation")
	//	.SetParam("act_type", "relu")
	//	.SetInput("data", fc2)
	//	.CreateSymbol(name + "_discriminator_relu_h2");
	//
	//return mxnet::cpp::LinearRegressionOutput(name + "_discriminator_lro1", relu2, data_label);
}

mxnet::cpp::Symbol MyResNetGAN(
	const std::string &name,
	int code_vector_length,
	int num_filter = 16,
	mx_float bn_momentum = 0.9
){
	mxnet::cpp::Symbol data/*rand*/ = mxnet::cpp::Symbol::Variable(/*"rand"*/"data");
	//mxnet::cpp::Symbol phase = mxnet::cpp::Symbol::Variable("phase");	//Определяет обучаем сейчас генератор(1)(фиксируем дискриминатор) или обучаем дискриминатор(0)(фиксируем генератор)
	mxnet::cpp::Symbol data_label = mxnet::cpp::Symbol::Variable("label");

	//mxnet::cpp::Symbol ggamma(name + "_generator_gamma");
	//mxnet::cpp::Symbol gbeta(name + "_generator_beta");
	//mxnet::cpp::Symbol gmmean(name + "_generator_mmean");
	//mxnet::cpp::Symbol gmvar(name + "_generator_mvar");

	//mxnet::cpp::Symbol gzscore = BatchNorm(name + "_generator_zscore", data, ggamma, gbeta, gmmean, gmvar, 0.001, bn_momentum);

	mxnet::cpp::Symbol generator = Decoder(name + "_generator", data, code_vector_length, num_filter, bn_momentum);
	
	//mxnet::cpp::Symbol dgamma(name + "_discriminator_gamma");
	//mxnet::cpp::Symbol dbeta(name + "_discriminator_beta");
	//mxnet::cpp::Symbol dmmean(name + "_discriminator_mmean");
	//mxnet::cpp::Symbol dmvar(name + "_discriminator_mvar");

	//mxnet::cpp::Symbol dzscore = BatchNorm(name + "_discriminator_zscore", generator, dgamma, dbeta, dmmean, dmvar, 0.001, bn_momentum);


	mxnet::cpp::Symbol discriminator = Encoder(name + "_discriminator", generator, code_vector_length, num_filter, bn_momentum);

	//mxnet::cpp::Symbol flat = mxnet::cpp::Flatten(name + "_discriminator_flatten", discriminator);

	auto fc = mxnet::cpp::Operator("FullyConnected")
		.SetParam("num_hidden", 2)
		.SetParam("no_bias", false)
		.SetInput("data", discriminator)
		.CreateSymbol(name + "_discriminator_fc_h");

	auto relu = mxnet::cpp::Operator("Activation")
		.SetParam("act_type", "relu")
		.SetInput("data", fc)
		.CreateSymbol(name + "_discriminator_relu_h");

	//auto sm = softmax(name + "_discriminator_sm_h", relu, 1);

	return SoftmaxOutput(name + "_discriminator_softmax2", relu/*see arXiv:1406.2661v1*/, data_label*(-1) + 1, 1.F, -1.F, false, false, false, mxnet::cpp::SoftmaxOutputNormalization::kBatch);

	//arXiv:1406.2661v1
	//In practice, equation 1 may not provide sufficient gradient for G to learn well.Early in learning,
	//when G is poor, D can reject samples with high confidence because they are clearly different from
	//the training data.In this case, log(1 − D(G(z))) saturates.Rather than training G to minimize
	//log(1 − D(G(z))) we can train G to maximize log D(G(z)).This objective function results in the
	//same fixed point of the dynamics of Gand D but provides much stronger gradients early in learning.

	//auto reshaped = mxnet::cpp::Reshape(name + "_discriminator_pool2_reshape", relu, mxnet::cpp::Shape(BATCH_SIZE, 2, 1, 1));
	//mxnet::cpp::Symbol m = mxnet::cpp::Pooling(name + "_discriminator_pool2", reshaped, mxnet::cpp::Shape(1, 2, 1, 1), mxnet::cpp::PoolingPoolType::kMax,
	//	false, false, mxnet::cpp::PoolingPoolingConvention::kValid/*, mxnet::cpp::Shape(1, 2, 1, 1)*/);
	//auto reshaped2 = mxnet::cpp::Reshape(name + "_discriminator_pool2_reshape2", m, mxnet::cpp::Shape(BATCH_SIZE));
	//return mxnet::cpp::MakeLoss(name + "_discriminator_loss2", mxnet::cpp::log(reshaped2 + mxnet::cpp::abs(reshaped2) + 1e-6)) * (-1);


	//auto fc2 = mxnet::cpp::Operator("FullyConnected")
	//	.SetParam("num_hidden", 1)
	//	.SetParam("no_bias", true)
	//	.SetInput("data", relu)
	//	.CreateSymbol(name + "_discriminator_fc_h2");

	//auto relu2 = mxnet::cpp::Operator("Activation")
	//	.SetParam("act_type", "relu")
	//	.SetInput("data", fc2)
	//	.CreateSymbol(name + "_discriminator_relu_h2");

	//return mxnet::cpp::LinearRegressionOutput(name + "_discriminator_lro2", relu2, data_label * (-1) + 1);
}

int main(int argc, char const* argv[]) 
{
	TRandomInt::Instance().Initialize((unsigned long)std::time(0));
	TRandomDouble::Instance().Initialize(RandomInt());

	try {
		//TRAIN

		//TEST

		//Dataset
		std::cout<<"dataset initialization..."<<std::endl;
		TFacesDatasetProperties fd_properties = FACES_DATASET_PROPERTIES_DEFAULTS;
		fd_properties.DatasetProperties.Dir = DATA_DIR;
		fd_properties.PoseModelFilename = POS_MODEL_FILENAME;
		fd_properties.DatasetProperties.ImgSize = cv::Size(DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE);
		fd_properties.DatasetProperties.UseMultiThreading = true;
		fd_properties.DatasetProperties.Warp = false;
		TFacesDataset train_dataset(fd_properties);
		std::cout<<"class number: "<< train_dataset.ClassNumber()<<std::endl;

		TFacesDatasetProperties ftd_properties = FACES_DATASET_PROPERTIES_DEFAULTS;
		fd_properties.DatasetProperties.Dir = DATA_DIR;
		fd_properties.PoseModelFilename = POS_MODEL_FILENAME;
		fd_properties.DatasetProperties.ImgSize = cv::Size(DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE);
		fd_properties.DatasetProperties.UseMultiThreading = true;
		fd_properties.DatasetProperties.Warp = false;
		TFacesDataset test_dataset(fd_properties);
		std::cout << "class number: " << test_dataset.ClassNumber() << std::endl;

		////Dataset
		//std::cout << "dataset initialization..." << std::endl;
		//TMNISTDatasetProperties md_properties;
		//md_properties.ImagePath =  "../train-images.idx3-ubyte";
		//md_properties.LabelPath = 	"../train-labels.idx1-ubyte";
		//md_properties.ImgSize = cv::Size(DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE);
		//TMNISTDataset train_dataset(md_properties);

		//TMNISTDatasetProperties mtd_properties;
		//md_properties.ImagePath =  "../t10k-images.idx3-ubyte";
		//md_properties.LabelPath = 	"../t10k-labels.idx1-ubyte";
		//md_properties.ImgSize = cv::Size(DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE);
		//TMNISTDataset test_dataset(mtd_properties);
		//std::cout << "class number: " << train_dataset.ClassNumber() << std::endl;

		//Model
		//auto resnet = MyResNetAutoEncoder(""/*"autoencoder"*/, FR_DESCRIPTOR_SIZE, 16, 0.9f);
		//resnet.Save("resnet.txt");

		auto generator = MyResNetGenerator("", FR_DESCRIPTOR_SIZE, 16, 0.9f);
		auto discriminator = MyResNetDiscriminator("", FR_DESCRIPTOR_SIZE, 16, 0.9f);
		auto gan = MyResNetGAN("", FR_DESCRIPTOR_SIZE, 16, 0.9f);
		gan.Save("gan.txt");
		generator.Save("gen.txt");
		discriminator.Save("discr.txt");



		TMxDNNTrainerProperties properties;
		properties.ExecutorProperties.NetImageWidth = NET_IMAGE_SIZE;
		properties.ExecutorProperties.NetImageHeight = NET_IMAGE_SIZE;
		properties.ExecutorProperties.NetImageChannels = 1;
		properties.ExecutorProperties.LabelChannels = 1;
		properties.ExecutorProperties.ModelFileName = MODEL_FILENAME;
		properties.ExecutorProperties.BatchSize = BATCH_SIZE;
		properties.ExecutorProperties.CodeVectorLength = FR_DESCRIPTOR_SIZE;
		properties.ExecutorProperties.ExecuteMode = DNNExecuteMode::AUTOENCODER;		//DNNExecuteMode::CLASSIFIER;//DNNExecuteMode::AUTOENCODER; //DNNExecuteMode::GAN
		properties.NBatchStatistics = 100;
		properties.TestBatches = 200;
		properties.OutputInfoTime = 1;
		properties.MaxEpoch = MAX_EPOCH;
		properties.StartLearningRate = START_LEARNING_RATE;
		properties.FinalLearningRate = 1e-6f;
		properties.EpochStep = 10000;
		properties.WeightDecay = WEIGHT_DECAY;
		properties.Err = ERR;
		properties.StateFileName = STATE_FILENAME;
		properties.TrainerOptimizer = TMxDNNTrainerOptimizer::ADAM;
	
		//TMxDNNTrainer trainer(properties, &train_dataset, &test_dataset, &resnet);
		TMxGANTrainer trainer(properties, &train_dataset, &test_dataset, &gan, &generator, &discriminator);
		trainer.Train();
	}	catch (std::exception &e) {
		dmlc::Error * E = dynamic_cast<dmlc::Error *>(&e);
		if (E) {
			LG << "Status: FAIL";
			LG << "With Error: " << MXGetLastError();
		} else {
			std::cout << e.what() << std::endl;
			return 1;
		}
	}
		return 0;
}