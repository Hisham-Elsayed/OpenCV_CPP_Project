#include "yolo.h"
#include "type.h"
#include "imageDetection.h"
#include "videoDetection.h"
#include "cameraDetection.h"

#include <memory>

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);  //silence logs

    string classesFile = "/yolo/coco.names";
    string modelConfiguration = "/yolo/yolov3.cfg";
    string modelWeights = "/yolo/yolov3.weights";

   

    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    std::vector<std::unique_ptr<Type>> detectors;

    // Add detectors (only if files exist or flags are set)
    detectors.push_back(std::make_unique<ImageDetector>("/Hisham/YOLO_test/YOLO_practice/img.jpg"));
    detectors.push_back(std::make_unique<VideoDetector>("/Hisham/YOLO_test/YOLO_practice/object_detection_test.mp4"));
    detectors.push_back(std::make_unique<CameraDetector>(0)); 

    // Loop through each and run detection
    for (auto& detector : detectors) {
        detector->detect(net, classes);
    }

    

 
    // cam.setCam(0);
    // cam.detectWebcam(cam.getCam(),net,classes);
    
    destroyAllWindows();
    return 0;
}
