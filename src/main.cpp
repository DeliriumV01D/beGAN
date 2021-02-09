//#define DO_NOT_USE_DLIB

#include <chrono>
#include <string>
#include <fstream>
#include <vector>
#include <filesystem>

#include "TMnistDataset.h"
#include "TMxDNNTrainer.h"
#include "CommonDefinitions.h"
#include "TDataset.h"
#include "TFacesDataset.h"

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
const int MAX_EPOCH = 1000;
const float START_LEARNING_RATE = 1e-4;
const float WEIGHT_DECAY = 1e-6;	//1e-6;//0.;
const float ERR = 1.e-5;
const int BATCH_SIZE = 32;

static const std::string DATA_DIR = "D:\\Delirium\\PROJECTS\\CheckPoint Smart\\FaceRecognizer\\FaceRecognizer\\Data";
static const std::string TEST_DATA_DIR = "D:\\Delirium\\PROJECTS\\CheckPoint Smart\\FaceRecognizer\\FaceRecognizer\\Test";
static const bool USE_DATA_AUGMENTATION = true;
//static const cv::Size DATASET_IMAGE_SIZE = { 96, 96 };
static const std::string POS_MODEL_FILENAME = "D:\\Delirium\\PROJECTS\\CheckPoint Smart\\FaceRecognizer\\FaceRecognizer\\FaceRecognizer\\shape_predictor_68_face_landmarks.dat";
static const std::string MODEL_FILENAME = "resnet.dat";
static const std::string STATE_FILENAME = "resnet_sync";
static const int FR_DESCRIPTOR_SIZE = 160;

mxnet::cpp::Symbol Decoder(
	const std::string &name,
	mxnet::cpp::Symbol data,
	int code_vector_length,
	int num_filter = 16,
	mx_float bn_momentum = 0.9
){
	auto input_shape = mxnet::cpp::Shape(num_filter * 32, 3, 3);

	auto fc1 = mxnet::cpp::Operator("FullyConnected")
		.SetParam("num_hidden", input_shape.Size())					//размер внутри батча
		.SetParam("no_bias", false)
		.SetInput("data", data)
		.CreateSymbol(name + "_fc1");
	auto relu1 = mxnet::cpp::Operator("Activation")
		.SetParam("act_type", "relu")
		.SetInput("data", fc1)
		.CreateSymbol(name + "_relu1");

	//Размер всех батчей
	auto reshaped = mxnet::cpp::Reshape(name + "_reshape", relu1, mxnet::cpp::Shape(BATCH_SIZE, num_filter * 32, 3, 3));  

	//mxnet::cpp::Symbol conv2 = GetTrConv(name + "_conv2", reshaped, num_filter * 64, mxnet::cpp::Shape(3, 3), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(0, 0), true, bn_momentum);  //столбцы [nbatches, 1024, 1, 1] -> [nbatches, 512, 3, 3]

	mxnet::cpp::Symbol block12 = MakeTrBlock(name + "_block12", reshaped, num_filter * 32, false, bn_momentum);																						//[nbatches, 512, 3, 3]
	mxnet::cpp::Symbol block11 = MakeTrBlock(name + "_block11", block12, num_filter * 32, true, bn_momentum);
	
	mxnet::cpp::Symbol block10 = MakeTrBlock(name + "_block10", block11, num_filter * 16, false, bn_momentum);																					//[nbatches, 256, 6, 6]
	mxnet::cpp::Symbol block9 = MakeTrBlock(name + "_block9", block10, num_filter * 16, true, bn_momentum);

	mxnet::cpp::Symbol block8 = MakeTrBlock(name + "_block8", block9, num_filter * 8, false, bn_momentum);
	mxnet::cpp::Symbol block7 = MakeTrBlock(name + "_block7", block8, num_filter * 8, true, bn_momentum);

	mxnet::cpp::Symbol block5 = MakeTrBlock(name + "_block5", block7, num_filter * 4, false, bn_momentum);
	mxnet::cpp::Symbol block4 = MakeTrBlock(name + "_block4", block5, num_filter * 4, true, bn_momentum);

	mxnet::cpp::Symbol block2 = MakeTrBlock(name + "_block2", block4, num_filter * 2, false, bn_momentum);
	mxnet::cpp::Symbol block1 = MakeTrBlock(name + "_block1", block2, num_filter * 2, true, bn_momentum);
	////mxnet::cpp::Symbol conv1 = GetTrConv(name + "_conv1", block1, num_filter, mxnet::cpp::Shape(5, 5), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(2, 2), true, bn_momentum);
	mxnet::cpp::Symbol conv0 = GetTrConv(name + "_conv0", block1, num_filter, mxnet::cpp::Shape(5, 5), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(2, 2), true, bn_momentum);

	//Здесь сложить все фильтры, возможно построить какой-то более сложный ансамбль  [nb, 16, 96, 96] -> [nb, 1, 96, 96]
	mxnet::cpp::Symbol sum = mxnet::cpp::sum(name + "_channel_sum", conv0, dmlc::optional<mxnet::cpp::Shape>(mxnet::cpp::Shape(1)), true );

	return sum;
}

mxnet::cpp::Symbol Encoder(
	const std::string &name,
	mxnet::cpp::Symbol data,
	int code_vector_length,
	int num_filter = 16,
	mx_float bn_momentum = 0.9
){
	mxnet::cpp::Symbol conv0 = GetConv(name + "_conv0", data, num_filter, mxnet::cpp::Shape(5, 5), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(2, 2), true, bn_momentum);
	//mxnet::cpp::Symbol conv1 = GetConv(name + "_conv1", conv0, num_filter, mxnet::cpp::Shape(5, 5), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(2, 2), true, bn_momentum);
	mxnet::cpp::Symbol block1 = MakeBlock(name + "_block1", conv0, num_filter, true, bn_momentum);
	mxnet::cpp::Symbol block2 = MakeBlock(name + "_block2", block1, num_filter * 2, false, bn_momentum);

	mxnet::cpp::Symbol block4 = MakeBlock(name + "_block4", block2, num_filter * 2, true, bn_momentum);
	mxnet::cpp::Symbol block5 = MakeBlock(name + "_block5", block4, num_filter * 4, false, bn_momentum);

	mxnet::cpp::Symbol block7 = MakeBlock(name + "_block7", block5, num_filter * 4, true, bn_momentum);
	mxnet::cpp::Symbol block8 = MakeBlock(name + "_block8", block7, num_filter * 8, false, bn_momentum);

	mxnet::cpp::Symbol block9 = MakeBlock(name + "_block9", block8, num_filter * 8, true, bn_momentum);
	mxnet::cpp::Symbol block10 = MakeBlock(name + "_block10", block9, num_filter * 16, false, bn_momentum);		//[nbatches, 256, 6, 6]
	
	mxnet::cpp::Symbol block11 = MakeBlock(name + "_block11", block10, num_filter * 16, true, bn_momentum);
	mxnet::cpp::Symbol block12 = MakeBlock(name + "_block12", block11, num_filter * 32, false, bn_momentum);		//[nbatches, 512, 3, 3]

	//mxnet::cpp::Symbol conv2 = GetConv(name + "_conv2", block12, num_filter * 64, mxnet::cpp::Shape(3, 3), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(0, 0), true, bn_momentum);  //столбцы [nbatches, 1024, 1, 1]
	
	////mxnet::cpp::Symbol pool = Pooling(name + "pool", body, pool_kernel, mxnet::cpp::PoolingPoolType::kAvg);
	//mxnet::cpp::Symbol flat = mxnet::cpp::Flatten(name + "_flatten", block9);

	auto fc2 = mxnet::cpp::Operator("FullyConnected")
		.SetParam("num_hidden", code_vector_length)
		.SetParam("no_bias", false)
		.SetInput("data", block12)
		.CreateSymbol(name + "_fc2");

	auto relu2 = mxnet::cpp::Operator("Activation")
		.SetParam("act_type", "relu")
		.SetInput("data", fc2)
		.CreateSymbol(name + "_relu2");

	return relu2;																																	//[nb, code_vector_length]
	////mxnet::cpp::Symbol fused = shortcut + conv2;
	////return Activation(name + "_relu", fused, "relu");
}

mxnet::cpp::Symbol MyResNetSymbol(
	const std::string &name,
	int num_class,
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


	mxnet::cpp::Symbol encoder = Encoder(name + "_encoder", zscore, code_vector_length, num_filter, bn_momentum);
	
	mxnet::cpp::Symbol fc_w(name + "_fc_w"), fc_b(name + "_fc_b");
	mxnet::cpp::Symbol fc = mxnet::cpp::FullyConnected(name + "_fc", encoder, fc_w, fc_b, num_class);

	////auto relu = mxnet::cpp::Operator("Activation")
	////							.SetParam("act_type", "relu")
	////							.SetInput("data", fc)
	////							.CreateSymbol(name + "_relu");
	//mxnet::cpp::Symbol fc2_w("fc2_w"), fc2_b("fc2_b");
	//mxnet::cpp::Symbol fc2 = mxnet::cpp::FullyConnected("fc2", fc, fc2_w, fc2_b, 1);
	//mxnet::cpp::Symbol am = mxnet::cpp::argmax(fc, dmlc::optional<int>(1));
	//mxnet::cpp::Shape shp = mxnet::cpp::Shape(BATCH_SIZE);	//mxnet::cpp::Reshape(fc2, shp)
	//mxnet::cpp::Symbol cross_entropy = data_label * log(am) + (1 - data_label) * log(1 - log(am));
	//return MakeLoss(0-cross_entropy);

	//fc1 < -mx.symbol.FullyConnected(data, num_hidden = 14, name = "fc1")
	//tanh1 < -mx.symbol.Activation(fc1, act_type = "tanh", name = "tanh1")
	//fc2 < -mx.symbol.FullyConnected(tanh1, num_hidden = 1, name = "fc2")
	//lro2 <- mx.symbol.MakeLoss(mx.symbol.square(mx.symbol.Reshape(fc2, shape = 0) - label), name="lro2")

	return SoftmaxOutput(name + "_softmax", fc, data_label);
	//return mxnet::cpp::LinearRegressionOutput(name + "_lro", fc, data_label);
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

	mxnet::cpp::Symbol encoder = Encoder(name + "_encoder", zscore, code_vector_length, num_filter, bn_momentum);
	mxnet::cpp::Symbol decoder = Decoder(name + "_decoder", encoder, code_vector_length, num_filter, bn_momentum);

	//return mxnet::cpp::LinearRegressionOutput(name + "_lro", decoder, data_label);
	return mxnet::cpp::MAERegressionOutput(name + "_mae_ro", decoder, data_label);
}



int main(int argc, char const* argv[]) 
{
	TRandomInt::Instance().Initialize(std::time(0));

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
		TFacesDataset dataset(fd_properties);
		std::cout<<"class number: "<<dataset.ClassNumber()<<std::endl;

		////Dataset
		//std::cout << "dataset initialization..." << std::endl;
		//TMNISTDatasetProperties md_properties;
		//md_properties.ImagePath =  "../train-images.idx3-ubyte";
		//md_properties.LabelPath = 	"../train-labels.idx1-ubyte";
		//md_properties.ImgSize = cv::Size(DATASET_IMAGE_SIZE, DATASET_IMAGE_SIZE);
		//TMNISTDataset dataset(md_properties);
		//std::cout << "class number: " << dataset.ClassNumber() << std::endl;

		//Model
		auto resnet = //LenetSymbol(dataset.ClassNumber());
									//ResNetSymbol(dataset.ClassNumber(), 3, 3, 16, 0.9, mxnet::cpp::Shape(2, 2));
									//MyResNetSymbol("resnet", dataset.ClassNumber(), FR_DESCRIPTOR_SIZE, 16, 0.9);
									MyResNetAutoEncoder("autoencoder", FR_DESCRIPTOR_SIZE, 16, 0.9);

		resnet.Save("resnet.txt");

		TMxDNNTrainerProperties properties;
		properties.DatasetImageWidth = DATASET_IMAGE_SIZE;
		properties.DatasetImageHeight = DATASET_IMAGE_SIZE;
		properties.NetImageWidth = NET_IMAGE_SIZE;
		properties.NetImageHeight = NET_IMAGE_SIZE;
		properties.MaxEpoch = MAX_EPOCH;
		properties.BatchSize = BATCH_SIZE;
		properties.StartLearningRate = START_LEARNING_RATE;
		properties.FinalLearningRate = 1e-6;
		properties.EpochStep = 15;
		properties.WeightDecay = WEIGHT_DECAY;
		properties.Err = ERR;
		properties.ModelFileName = MODEL_FILENAME;
		properties.StateFileName = STATE_FILENAME;
		properties.TrainerOptimizer = TMxDNNTrainerOptimizer::ADAM;
		properties.TrainerMode = TMxDNNTrainerMode::AUTOENCODER;		//TMxDNNTrainerMode::CLASSIFIER;//TMxDNNTrainerMode::AUTOENCODER;
		TMxDNNTrainer trainer(properties, &dataset, &resnet);
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