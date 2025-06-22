#include "yolo.h"



using namespace cv;
using namespace dnn;
using namespace std;

YoloDetector::YoloDetector(const std::string& class_file, const std::string& config_file, const std::string& weights_file) : classesFile(class_file),
                                                                                        modelConfiguration(config_file),
                                                                                        modelWeights(weights_file)   
{}

bool YoloDetector::load()
{
    // Check if YOLO files exist
    ifstream checkYolo(classesFile);
    if (!checkYolo) {
        cerr << "Failed to find YOLO files in: /yolo/" << endl;
        return false;
    }
   
    // Load class names from file
    classes.clear();
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Load YOLO network from configuration and weights
    net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    return !classes.empty();
}

Net& YoloDetector::getNet()
{
    return net;
}

vector<string> YoloDetector::getOutputsNames() const
{
    vector<string> names;
    vector<int> outLayers = net.getUnconnectedOutLayers();
    vector<string> layersNames = net.getLayerNames();
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    return names;
}

void YoloDetector::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame) {
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 3);
    string label = format("%.2f", conf);
    if (!classes.empty() && classId < static_cast<int>(classes.size()))
        label = classes[classId] + ":" + label;
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(), 1);
}


void YoloDetector::postprocess(Mat& frame, const vector<Mat>& outs) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    // Parse the outputs
    for (const Mat& out : outs) {
        for (int i = 0; i < out.rows; ++i) {
            Mat scores = out.row(i).colRange(5, out.cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = static_cast<int>(out.at<float>(i, 0) * frame.cols);
                int centerY = static_cast<int>(out.at<float>(i, 1) * frame.rows);
                int width   = static_cast<int>(out.at<float>(i, 2) * frame.cols);
                int height  = static_cast<int>(out.at<float>(i, 3) * frame.rows);
                int left    = centerX - width / 2;
                int top     = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non-maximum suppression and draw results
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (int idx : indices)
        drawPred(classIds[idx], confidences[idx], boxes[idx].x, boxes[idx].y,
                 boxes[idx].x + boxes[idx].width, boxes[idx].y + boxes[idx].height, frame);
}
