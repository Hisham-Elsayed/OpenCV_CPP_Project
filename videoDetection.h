#ifndef VIDEO_DETECTION_H
#define VIDEO_DETECTION_H

#include "type.h"

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
     */
    VideoDetector(const std::string &path);

    /**
     * @brief Detect objects in the video using the given network and class names.
     * @param net Reference to the loaded YOLO network.
     * @param classes Vector of class names.
     */
    void detect(cv::dnn::Net &net, const std::vector<std::string>& classes) override;

    private:
    // Path to the video file.
    std::string videoPath;
};


#endif