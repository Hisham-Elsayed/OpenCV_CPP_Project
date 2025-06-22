#ifndef IMAGE_DETECTION_H
#define IMAGE_DETECTION_H

#include "type.h"

/**
 * @class ImageDetector
 * @brief Performs object detection on a static image.
 */
class ImageDetector : public Type
{
    public:
    /**
     * @brief Constructor.
     * @param path Path to the image file.
     */
    ImageDetector(const std::string &path);

     /**
     * @brief Detect objects in the image using the given network and class names.
     * @param net Reference to the loaded YOLO network.
     * @param classes Vector of class names.
     */
    void detect(cv::dnn::Net& net, const std::vector<std::string>& classes) override;


    private:
    // Path to the image file.
    std::string imagePath;
};



#endif