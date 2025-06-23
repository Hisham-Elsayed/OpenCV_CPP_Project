#include "yolo.h"
#include "type.h"
#include "imageDetection.h"
#include "videoDetection.h"
#include "cameraDetection.h"
#include <memory>

using namespace cv;
using namespace dnn;
using namespace std;



/**
 * @brief Entry point for the YOLO Object Detection application.
 * 
 * This function demonstrates a modern, OOP-based approach to running YOLO object detection
 * on images, videos, and live camera streams. It initializes multiple YOLO models, loads class names,
 * and creates detector objects for each input source. Each detector composes a YoloDetector instance,
 * allowing for flexible use of different YOLO models per source. All detectors are managed polymorphically
 * via a common base class, enabling easy extension and maintenance.
 * 
 * @return int Returns 0 on successful execution.
 */
int main() {

    #if defined(_WIN32) || defined(_WIN64)
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);  // silence logs
    #endif

    // File paths for YOLO model and class names
    string classesFile = "/yolo/coco.names";

    string modelConfiguration = "/yolo/yolov4-tiny.cfg";
    string modelWeights = "/yolo/yolov4-tiny.weights";

    string model2 = "/yolo/yolov4.cfg";
    string weight2 ="/yolo/yolov4.weights";

    YoloDetector yolov4_tiny(classesFile,modelConfiguration,modelWeights);
    yolov4_tiny.load();

    YoloDetector yolov4(classesFile,model2,weight2);
    yolov4.load();

    // Create a vector of unique pointers to different detector types
    vector<unique_ptr<Type>> detectors;

    // Add detectors for image, video, and camera sources
    detectors.emplace_back(make_unique<ImageDetector>("../Samples/Images/dog_bike_car.jpg",yolov4));
    detectors.emplace_back(make_unique<VideoDetector>("../Samples/Videos/object_detection_test.mp4",yolov4_tiny));
    detectors.emplace_back(make_unique<VideoDetector>("../Samples/Videos/Vehicle Dataset Sample 2.mp4",yolov4_tiny));
    detectors.emplace_back(make_unique<CameraDetector>(0,yolov4_tiny)); 

    // Run detection for each detector
    for (auto& detector : detectors) {
        detector->detect();
    }


    // Close all OpenCV windows
    destroyAllWindows();
    
    return 0;
}
