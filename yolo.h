#ifndef YOLO_H
#define YOLO_H

#include "type.h"

extern float confThreshold;
extern float nmsThreshold;
extern int inpWidth;
extern int inpHeight;

std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, std::vector<std::string>& classes);
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, std::vector<std::string>& classes);




#endif 