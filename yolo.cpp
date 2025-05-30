#include "yolo.h"

float confThreshold = 0.5;
float nmsThreshold = 0.4;
int inpWidth = 416;
int inpHeight = 416;

using namespace cv;
using namespace dnn;
using namespace std;

vector<string> getOutputsNames(const Net& net) {
    static vector<string> names;
    if (names.empty()) {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<string> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, vector<string>& classes) {
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 3);
    string label = format("%.2f", conf);
    if (!classes.empty() && classId < (int)classes.size())
        label = classes[classId] + ":" + label;
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(), 1);
}

void postprocess(Mat& frame, const vector<Mat>& outs, Net& net, vector<string>& classes) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (const Mat& out : outs) {
        for (int i = 0; i < out.rows; ++i) {
            Mat scores = out.row(i).colRange(5, out.cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(out.at<float>(i, 0) * frame.cols);
                int centerY = (int)(out.at<float>(i, 1) * frame.rows);
                int width   = (int)(out.at<float>(i, 2) * frame.cols);
                int height  = (int)(out.at<float>(i, 3) * frame.rows);
                int left    = centerX - width / 2;
                int top     = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (int idx : indices)
        drawPred(classIds[idx], confidences[idx], boxes[idx].x, boxes[idx].y,
                 boxes[idx].x + boxes[idx].width, boxes[idx].y + boxes[idx].height, frame, classes);
}
