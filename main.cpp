#include "yolo.h"
#include "type.h"
#include "imageDetection.h"
#include "videoDetection.h"
#include "cameraDetection.h"

#include <memory>

#include <direct.h>

using namespace cv;
using namespace dnn;
using namespace std;



/**
 * @brief Entry point for the YOLO Object Detection application.
 * 
 * This function initializes the YOLO model, loads class names, and creates detector objects
 * for image, video, and camera sources. It then runs detection for each source.
 * 
 * @return int Returns 0 on successful execution.
 */
int main() {
    // Silence OpenCV logging output for cleaner console
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);  

    // File paths for YOLO model and class names
    string classesFile = "/yolo/coco.names";
    string modelConfiguration = "/yolo/yolov4-tiny.cfg";
    string modelWeights = "/yolo/yolov4-tiny.weights";

    // Check if YOLO files exist
    ifstream checkYolo(classesFile);
    if (!checkYolo) {
        cerr << "Failed to find YOLO files in: /yolo/" << endl;
        return 1;
    }
   
    // Load class names from file
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Load YOLO network from configuration and weights
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Create a vector of unique pointers to different detector types
    vector<unique_ptr<Type>> detectors;

    // Add detectors for image, video, and camera sources
    detectors.emplace_back(make_unique<ImageDetector>("../Samples/Images/dog_bike_car.jpg"));
    detectors.emplace_back(make_unique<VideoDetector>("../Samples/Videos/object_detection_test.mp4"));
    detectors.emplace_back(make_unique<VideoDetector>("../Samples/Videos/Vehicle Dataset Sample 2.mp4"));
    detectors.emplace_back(make_unique<CameraDetector>(0)); 

    // Run detection for each detector
    for (auto& detector : detectors) {
        detector->detect(net, classes);
    }

    cout << "Current working directory: " << _getcwd(NULL, 0) << endl;

    // Close all OpenCV windows
    destroyAllWindows();
    
    return 0;
}
