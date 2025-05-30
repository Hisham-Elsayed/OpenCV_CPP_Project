#ifndef YOLO_H
#define YOLO_H

#include "type.h"

extern float confThreshold;
extern float nmsThreshold;
extern int inpWidth;
extern int inpHeight;

vector<string> getOutputsNames(const Net& net);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, vector<string>& classes);
void postprocess(Mat& frame, const vector<Mat>& outs, Net& net, vector<string>& classes);




#endif 