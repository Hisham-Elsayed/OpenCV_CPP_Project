#ifndef YOLO_H
#define YOLO_H

#include "type.h"

/**
 * @file yolo.h
 * @brief Declarations for YOLO utility functions and parameters.
 */

extern float confThreshold; ///< Confidence threshold for filtering weak detections.
extern float nmsThreshold;  ///< Non-maximum suppression threshold.
extern int inpWidth;        ///< Width of network's input image.
extern int inpHeight;       ///< Height of network's input image.

/**
 * @brief Get the names of the output layers of the network.
 * @param net Reference to the loaded YOLO network.
 * @return Vector of output layer names.
 */
std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);

/**
 * @brief Draws a predicted bounding box with label on the frame.
 * @param classId Class index.
 * @param conf Confidence score.
 * @param left Left coordinate of bounding box.
 * @param top Top coordinate of bounding box.
 * @param right Right coordinate of bounding box.
 * @param bottom Bottom coordinate of bounding box.
 * @param frame Frame to draw on.
 * @param classes Vector of class names.
 */
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<std::string>& classes);

/**
 * @brief Process the network outputs and draw bounding boxes on the frame.
 * @param frame Frame to draw on.
 * @param outs Network outputs.
 * @param net Reference to the loaded YOLO network.
 * @param classes Vector of class names.
 */
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, std::vector<std::string>& classes);




#endif 