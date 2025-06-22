#ifndef IMAGE_DETECTION_H
#define IMAGE_DETECTION_H

#include "type.h"
#include "yolo.h"

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
     * @param yolo Reference to a YoloDetector instance to use for detection.
     */
    ImageDetector(const std::string &path, YoloDetector& yolo);

     /**
     * @brief Detect objects in the image using the given network and class names.
     */
    void detect() override;


    private:
    std::string imagePath;      // Path to the image file.
    YoloDetector& yolo;
};



#endif