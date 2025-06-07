#ifndef CAMERA_DETECTION
#define CAMERA_DETECTION

#include "type.h"

/**
 * @class CameraDetector
 * @brief Performs object detection using a live camera feed.
 */
class CameraDetector : public Type
{
    public:
    /**
     * @brief Constructor.
     * @param path Camera index (usually 0 for default camera).
     */
    CameraDetector(const int &path);

    /**
     * @brief Detect objects from the camera using the given network and class names.
     * @param net Reference to the loaded YOLO network.
     * @param classes Vector of class names.
     */
    void detect(cv::dnn::Net &net, std::vector<std::string> &classes) override;

    private:
    // Camera index.
    int cam;

};

#endif