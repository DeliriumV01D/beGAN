#ifndef TFACE_DETECTOR_H
#define TFACE_DETECTOR_H

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <WriteToLog.h>
#include <vector>

static const dlib::uint32	INNER_EYES_AND_BOTTOM_LIP[] = { 39, 42, 57 },
										OUTER_EYES_AND_NOSE[] = { 36, 45, 33 };

static const cv::Point2f TEMPLATE[] = {
			cv::Point2f(0.0792396913815, 0.339223741112), cv::Point2f(0.0829219487236, 0.456955367943),
			cv::Point2f(0.0967927109165, 0.575648016728), cv::Point2f(0.122141515615, 0.691921601066),
			cv::Point2f(0.168687863544, 0.800341263616), cv::Point2f(0.239789390707, 0.895732504778),
			cv::Point2f(0.325662452515, 0.977068762493), cv::Point2f(0.422318282013, 1.04329000149),
			cv::Point2f(0.531777802068, 1.06080371126), cv::Point2f(0.641296298053, 1.03981924107),
			cv::Point2f(0.738105872266, 0.972268833998), cv::Point2f(0.824444363295, 0.889624082279),
			cv::Point2f(0.894792677532, 0.792494155836), cv::Point2f(0.939395486253, 0.681546643421),
			cv::Point2f(0.96111933829, 0.562238253072), cv::Point2f(0.970579841181, 0.441758925744),
			cv::Point2f(0.971193274221, 0.322118743967), cv::Point2f(0.163846223133, 0.249151738053),
			cv::Point2f(0.21780354657, 0.204255863861), cv::Point2f(0.291299351124, 0.192367318323),
			cv::Point2f(0.367460241458, 0.203582210627), cv::Point2f(0.4392945113, 0.233135599851),
			cv::Point2f(0.586445962425, 0.228141644834), cv::Point2f(0.660152671635, 0.195923841854),
			cv::Point2f(0.737466449096, 0.182360984545), cv::Point2f(0.813236546239, 0.192828009114),
			cv::Point2f(0.8707571886, 0.235293377042), cv::Point2f(0.51534533827, 0.31863546193),
			cv::Point2f(0.516221448289, 0.396200446263), cv::Point2f(0.517118861835, 0.473797687758),
			cv::Point2f(0.51816430343, 0.553157797772), cv::Point2f(0.433701156035, 0.604054457668),
			cv::Point2f(0.475501237769, 0.62076344024), cv::Point2f(0.520712933176, 0.634268222208),
			cv::Point2f(0.565874114041, 0.618796581487), cv::Point2f(0.607054002672, 0.60157671656),
			cv::Point2f(0.252418718401, 0.331052263829), cv::Point2f(0.298663015648, 0.302646354002),
			cv::Point2f(0.355749724218, 0.303020650651), cv::Point2f(0.403718978315, 0.33867711083),
			cv::Point2f(0.352507175597, 0.349987615384), cv::Point2f(0.296791759886, 0.350478978225),
			cv::Point2f(0.631326076346, 0.334136672344), cv::Point2f(0.679073381078, 0.29645404267),
			cv::Point2f(0.73597236153, 0.294721285802), cv::Point2f(0.782865376271, 0.321305281656),
			cv::Point2f(0.740312274764, 0.341849376713), cv::Point2f(0.68499850091, 0.343734332172),
			cv::Point2f(0.353167761422, 0.746189164237), cv::Point2f(0.414587777921, 0.719053835073),
			cv::Point2f(0.477677654595, 0.706835892494), cv::Point2f(0.522732900812, 0.717092275768),
			cv::Point2f(0.569832064287, 0.705414478982), cv::Point2f(0.635195811927, 0.71565572516),
			cv::Point2f(0.69951672331, 0.739419187253), cv::Point2f(0.639447159575, 0.805236879972),
			cv::Point2f(0.576410514055, 0.835436670169), cv::Point2f(0.525398405766, 0.841706377792),
			cv::Point2f(0.47641545769, 0.837505914975), cv::Point2f(0.41379548902, 0.810045601727),
			cv::Point2f(0.380084785646, 0.749979603086), cv::Point2f(0.477955996282, 0.74513234612),
			cv::Point2f(0.523389793327, 0.748924302636), cv::Point2f(0.571057789237, 0.74332894691),
			cv::Point2f(0.672409137852, 0.744177032192), cv::Point2f(0.572539621444, 0.776609286626),
			cv::Point2f(0.5240106503, 0.783370783245), cv::Point2f(0.477561227414, 0.778476346951)};


///Класс для детектирования лиц на изображении и ректификации
class TFaceDetector {
protected:
	dlib::frontal_face_detector Detector;
	dlib::shape_predictor PoseModel;
	std::mutex Mutex;
public:
	TFaceDetector(const std::string &pose_model_filename);
	~TFaceDetector(){};
	
	//Detect faces in image
	//This face detector is made using the now classic Histogram of Oriented
	//Gradients (HOG) feature combined with a linear classifier, an image pyramid, and sliding window detection scheme.
	std::vector<dlib::rectangle> Detect(const cv::Mat &img);

	//Extracting copies of each face that are cropped, rotated upright, and scaled to a standard size
	//The pose estimator was created by using dlib's implementation of the paper:
	//One Millisecond Face Alignment with an Ensemble of Regression Trees by
	//Vahid Kazemi and Josephine Sullivan, CVPR 2014 and was trained on the iBUG 300-W face landmark dataset.
	void ExtractFaces(const cv::Mat &img, const std::vector<dlib::rectangle> &face_rects, dlib::array<dlib::matrix<unsigned char> > &faces) //dlib::array<dlib::array2d<dlib::rgb_pixel> > &faces);
	{
		dlib::cv_image<unsigned char> cimg(img); //Note that this just wraps the Mat object, it doesn't copy anything.
		//dlib::array<dlib::array2d<dlib::rgb_pixel> > result;
		std::vector<dlib::full_object_detection> shapes;
		for (unsigned long i = 0; i < face_rects.size(); ++i)
		{
			//Find the pose of each face. The pose takes the form of 68 landmarks.  These are
			//points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.
			shapes.push_back(PoseModel(cimg, face_rects[i]));
		}

		dlib::extract_image_chips(cimg, get_face_chip_details(shapes), faces);		
	};
};

inline TFaceDetector ::	TFaceDetector(const std::string &pose_model_filename)
{
	//Load face detection and pose estimation models.
	Detector = dlib::get_frontal_face_detector();
	dlib::deserialize(pose_model_filename) >> PoseModel;
};

//Detect faces in image
//This face detector is made using the now classic Histogram of Oriented
//Gradients (HOG) feature combined with a linear classifier, an image pyramid, and sliding window detection scheme.
inline std::vector<dlib::rectangle> TFaceDetector :: Detect(const cv::Mat &img)
{
	dlib::cv_image<unsigned char> cimg(img);	//<dlib::bgr_pixel> cimg(img);//Note that this just wraps the Mat object, it doesn't copy anything.
	std::vector <dlib::rectangle> result;
	Mutex.lock();
	try {
		result = Detector(cimg);
	} catch (std::exception &e) {
		Mutex.unlock();
		TException * E = dynamic_cast<TException *>(&e);
		if (E) throw (*E);
		else throw e;
	};
	Mutex.unlock();
	return result;
};

#endif