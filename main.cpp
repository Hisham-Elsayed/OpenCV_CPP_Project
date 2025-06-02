#include "yolo.h"
#include "type.h"
#include "imageDetection.h"
#include "videoDetection.h"
#include "cameraDetection.h"

#include <memory>

using namespace cv;
using namespace dnn;
using namespace std;

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);  //silence logs

    string classesFile = "/yolo/coco.names";
    string modelConfiguration = "/yolo/yolov4-tiny.cfg";
    string modelWeights = "/yolo/yolov4-tiny.weights";

   

    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);


    vector<unique_ptr<Type>> detectors;

    detectors.push_back(make_unique<ImageDetector>("/Hisham/YOLO_test/YOLO_practice/img.jpg"));
    detectors.push_back(make_unique<VideoDetector>("/Hisham/YOLO_test/YOLO_practice/object_detection_test.mp4"));
    detectors.push_back(make_unique<CameraDetector>(0)); 

    for (auto& detector : detectors) {
        detector->detect(net, classes);
    }
 
    destroyAllWindows();
    return 0;
}
