#ifndef YOLO_H
#define YOLO_H
/**
 * @file yolo.h
 * @brief Declaration of the YoloDetector class, which encapsulates YOLO network loading, configuration, and detection utilities.
 */

#include "type.h"
#include <string>
#include <vector>



 /**
 * @class YoloDetector
 * @brief Encapsulates YOLO network loading, configuration, and detection utilities.
 */
class YoloDetector
{
    public:
    /**
    * @brief Constructs a YoloDetector with the given file paths.
    * @param class_file Path to class names file.
    * @param config_file Path to YOLO configuration file.
    * @param weights_file Path to YOLO weights file.
    */
    YoloDetector(const std::string& class_file, const std::string& config_file, const std::string& weights_file);

    /**
    * @brief Loads the YOLO network and class names.
    * @return True if loading is successful, false otherwise.
    */
    bool load();

    /**
    * @brief Provides access to the loaded YOLO network.
    * @return Reference to the internal cv::dnn::Net object.
    */
    cv::dnn::Net& getNet();

    /**
    * @brief Get the names of the output layers of the network.
    * @return Vector of output layer names.
    */
    std::vector<std::string> getOutputsNames() const;

    /**
    * @brief Draws a predicted bounding box with label on the frame.
    * @param classId Class index.
    * @param conf Confidence score.
    * @param left Left coordinate of bounding box.
    * @param top Top coordinate of bounding box.
    * @param right Right coordinate of bounding box.
    * @param bottom Bottom coordinate of bounding box.
    * @param frame Frame to draw on.
    */
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

    /**
    * @brief Process the network outputs and draw bounding boxes on the frame.
    * @param frame Frame to draw on.
    * @param outs Network outputs.
    */
    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs);

    static constexpr float confThreshold = 0.5f; ///< Confidence threshold for filtering weak detections.
    static constexpr float nmsThreshold = 0.4f;  ///< Non-maximum suppression threshold.
    static constexpr int inpWidth = 416;        ///< Width of network's input image.
    static constexpr int inpHeight = 416;       ///< Height of network's input image.


    private:
    std::string classesFile;
    std::string modelConfiguration;
    std::string modelWeights;

    cv::dnn::Net net;
    std::vector<std::string> classes;
};

#endif 