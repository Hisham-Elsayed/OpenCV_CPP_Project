#ifndef TYPE_H
#define TYPE_H


#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <sstream>


/**
 * @class Type
 * @brief Abstract base class for all detection types (image, video, camera).
 * 
 * Provides a common interface for detection classes. 
 * All derived classes must implement the detect() method.
 */
class Type
{
    public:
    /**
     * @brief Perform object detection using the provided network and class names.
     * @param net Reference to the loaded YOLO network.
     * @param classes Vector of class names.
     */
    virtual void detect(cv::dnn::Net& net, std::vector<std::string>& classes) = 0; //pure virtual function

     /**
     * @brief Virtual destructor for safe polymorphic deletion.
     */
    virtual ~Type() = default; // virtual destructor

    protected:
    // Name of the window used for displaying results.
    std::string windowName;
};

#endif