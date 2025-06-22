#ifndef VIDEO_DETECTION_H
#define VIDEO_DETECTION_H

#include "type.h"
#include "yolo.h"

/**
 * @class VideoDetector
 * @brief Performs object detection on a video file.
 */
class VideoDetector : public Type
{
   public:
   /**
     * @brief Constructor.
     * @param path Path to the video file.
     * @param yolo Reference to a YoloDetector instance to use for detection.
     */
    VideoDetector(const std::string &path, YoloDetector& yolo);

    /**
     * @brief Detect objects in the video using the given network and class names.
     */
    void detect() override;

    private:
    std::string videoPath;      // Path to the video file.
    YoloDetector& yolo;
};


#endif