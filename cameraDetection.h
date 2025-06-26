#ifndef CAMERA_DETECTION
#define CAMERA_DETECTION

#include "type.h"
#include "yoloDetection.h"

/**
 * @class CameraDetector
 * @brief Performs object detection using a live camera feed.
 */
class CameraDetector : public Type
{
    public:
    /**
     * @brief Constructor.
     * @param camIndex Camera index (usually 0 for default camera).
     * @param yolo Reference to a YoloDetector instance to use for detection.
     */
    CameraDetector(const int &camIndex, YoloDetector& yolo);

    /**
     * @brief Detect objects from the camera using the given network and class names.
     */
    void detect() override;

    private:
    int cam;    // Camera index.
    YoloDetector& yolo;

};

#endif