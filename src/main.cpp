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
const float START_LEARNING_RATE = 1e-2;
const float WEIGHT_DECAY = 1e-6;
const float ERR = 1.e-5;
const int BATCH_SIZE = 64;

static const std::string DATA_DIR = "D:\\Delirium\\PROJECTS\\CheckPoint Smart\\FaceRecognizer\\FaceRecognizer\\Data";
static const std::string TEST_DATA_DIR = "D:\\Delirium\\PROJECTS\\CheckPoint Smart\\FaceRecognizer\\FaceRecognizer\\Test";
static const bool USE_DATA_AUGMENTATION = true;
//static const cv::Size DATASET_IMAGE_SIZE = { 96, 96 };
static const std::string POS_MODEL_FILENAME = "D:\\Delirium\\PROJECTS\\CheckPoint Smart\\FaceRecognizer\\FaceRecognizer\\FaceRecognizer\\shape_predictor_68_face_landmarks.dat";
static const std::string MODEL_FILENAME = "resnet.dat";
static const int FR_DESCRIPTOR_SIZE = 160;


mxnet::cpp::Symbol MyResNetSymbol(
	int num_class,
	int num_filter = 16,
	mx_float bn_momentum = 0.9
) {
	// data and label
	mxnet::cpp::Symbol data = mxnet::cpp::Symbol::Variable("data");
	mxnet::cpp::Symbol data_label = mxnet::cpp::Symbol::Variable("label");

	mxnet::cpp::Symbol gamma("gamma");
	mxnet::cpp::Symbol beta("beta");
	mxnet::cpp::Symbol mmean("mmean");
	mxnet::cpp::Symbol mvar("mvar");

	mxnet::cpp::Symbol zscore = BatchNorm("zscore", data, gamma, beta, mmean, mvar, 0.001, bn_momentum);

	mxnet::cpp::Symbol conv0 = GetConv("conv0", zscore, num_filter, mxnet::cpp::Shape(5, 5), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(1, 1), true, bn_momentum);
	mxnet::cpp::Symbol conv1 = GetConv("conv1", conv0, num_filter, mxnet::cpp::Shape(5, 5), mxnet::cpp::Shape(1, 1), mxnet::cpp::Shape(1, 1), true, bn_momentum);

	mxnet::cpp::Symbol block1 = MakeBlock("block1", conv1, num_filter, true, bn_momentum);
	mxnet::cpp::Symbol block2 = MakeBlock("block2", block1, num_filter * 2, false, bn_momentum);

	mxnet::cpp::Symbol block3 = MakeBlock("block3", block2, num_filter * 2, true, bn_momentum);
	mxnet::cpp::Symbol block4 = MakeBlock("block4", block3, num_filter * 2, true, bn_momentum);
	mxnet::cpp::Symbol block5 = MakeBlock("block5", block4, num_filter * 4, false, bn_momentum);

	mxnet::cpp::Symbol block6 = MakeBlock("block6", block5, num_filter * 4, true, bn_momentum);
	mxnet::cpp::Symbol block7 = MakeBlock("block7", block6, num_filter * 4, true, bn_momentum);
	mxnet::cpp::Symbol block8 = MakeBlock("block8", block7, num_filter * 4, true, bn_momentum);
	//mxnet::cpp::Symbol block9 = MakeBlock("block9", block8, num_filter * 4, true, bn_momentum);
	//mxnet::cpp::Symbol block10 = MakeBlock("block10", block9, num_filter * 4, true, bn_momentum);

	//mxnet::cpp::Symbol pool = Pooling("pool", body, pool_kernel, mxnet::cpp::PoolingPoolType::kAvg);

	mxnet::cpp::Symbol flat = mxnet::cpp::Flatten("flatten", block8);

	auto fc2 = mxnet::cpp::Operator("FullyConnected")
		.SetParam("num_hidden", FR_DESCRIPTOR_SIZE)															//!!!
		.SetParam("no_bias", false)
		.SetInput("data", flat)
		.CreateSymbol("fc2");
	auto relu2 = mxnet::cpp::Operator("Activation")
		.SetParam("act_type", "relu")
		.SetInput("data", fc2)
		.CreateSymbol("relu2");

	mxnet::cpp::Symbol fc_w("fc_w"), fc_b("fc_b");
	mxnet::cpp::Symbol fc = mxnet::cpp::FullyConnected("fc", relu2, fc_w, fc_b, num_class);

	return SoftmaxOutput("softmax", fc, data_label);
}




///!!!Принимает одноканальное изображение, поэтому в ResizeInput выкинул tile!!! И поменял data_shape
mxnet::cpp::Symbol LenetSymbol(int num_classes) {
	/*define the symbolic net*/
	mxnet::cpp::Symbol data = mxnet::cpp::Symbol::Variable("data");
	mxnet::cpp::Symbol data_label = mxnet::cpp::Symbol::Variable("label");
	mxnet::cpp::Symbol conv1_w("conv1_w"), conv1_b("conv1_b");
	mxnet::cpp::Symbol conv2_w("conv2_w"), conv2_b("conv2_b");
	mxnet::cpp::Symbol conv3_w("conv3_w"), conv3_b("conv3_b");
	mxnet::cpp::Symbol conv4_w("conv4_w"), conv4_b("conv4_b");
	mxnet::cpp::Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
	mxnet::cpp::Symbol fc2_w("fc2_w"), fc2_b("fc2_b");

	mxnet::cpp::Symbol conv1 = Convolution("conv1", data, conv1_w, conv1_b, mxnet::cpp::Shape(5, 5), 16);
	mxnet::cpp::Symbol tanh1 = Activation("tanh1", conv1, mxnet::cpp::ActivationActType::kTanh);
	mxnet::cpp::Symbol pool1 = Pooling("pool1", tanh1, mxnet::cpp::Shape(2, 2), mxnet::cpp::PoolingPoolType::kMax,
		false, false, mxnet::cpp::PoolingPoolingConvention::kValid, mxnet::cpp::Shape(2, 2));

	mxnet::cpp::Symbol conv2 = Convolution("conv2", pool1, conv2_w, conv2_b,	mxnet::cpp::Shape(5, 5), 32);
	mxnet::cpp::Symbol tanh2 = Activation("tanh2", conv2, mxnet::cpp::ActivationActType::kTanh);
	mxnet::cpp::Symbol pool2 = Pooling("pool2", tanh2, mxnet::cpp::Shape(2, 2), mxnet::cpp::PoolingPoolType::kMax,
		false, false, mxnet::cpp::PoolingPoolingConvention::kValid, mxnet::cpp::Shape(2, 2));

	mxnet::cpp::Symbol conv3 = Convolution("conv3", pool2, conv3_w, conv3_b, mxnet::cpp::Shape(2, 2), 64);
	mxnet::cpp::Symbol tanh3 = Activation("tanh3", conv3, mxnet::cpp::ActivationActType::kTanh);
	mxnet::cpp::Symbol pool3 = Pooling("pool3", tanh3, mxnet::cpp::Shape(2, 2), mxnet::cpp::PoolingPoolType::kMax,
		false, false, mxnet::cpp::PoolingPoolingConvention::kValid, mxnet::cpp::Shape(2, 2));

	mxnet::cpp::Symbol flatten = Flatten("flatten", pool3);
	mxnet::cpp::Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, FR_DESCRIPTOR_SIZE);			///!!!!
	mxnet::cpp::Symbol tanh5 = Activation("tanh5", fc1, mxnet::cpp::ActivationActType::kTanh);
	mxnet::cpp::Symbol fc2 = FullyConnected("fc2", tanh5, fc2_w, fc2_b, num_classes);					///!!!!

	mxnet::cpp::Symbol lenet = SoftmaxOutput("softmax", fc2, data_label);
	return lenet;
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
									ResNetSymbol(dataset.ClassNumber(), 3, 3, 16, 0.9, mxnet::cpp::Shape(2, 2));
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
		properties.TrainerOptimizer = TMxDNNTrainerOptimizer::ADAM;

		TMxDNNTrainer trainer(properties, &dataset, &resnet);
		trainer.Initialize();
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