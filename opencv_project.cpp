//#include "opencv_project.h"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace dnn;
using namespace std;



float confThreshold = 0.5;
float nmsThreshold = 0.4;
int inpWidth = 416;
int inpHeight = 416;

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

class Type
{
public:
void detectImage(const string& imagePath, Net& net, vector<string>& classes) {
    Mat image = imread(imagePath);

    if (image.empty()) {
        cerr << "Failed to load image from " << imagePath << endl;
        return;
    }

    // Create a 4D blob from the image
    Mat blob;
    blobFromImage(image, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

    // Set the input to the network
    net.setInput(blob);

    // Forward pass
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Process detections and draw bounding boxes
    postprocess(image, outs, net, classes);

    // Display the image
    imshow("YOLOv3 Image Detection", image);
    waitKey(0); // Wait for any key press to close the window
}

void detectWebcam(const int &cam, Net &net, vector<string> &classes)
{
    VideoCapture cap(cam);        //to use webcam
    if (!cap.isOpened()) {
        cerr << "Error opening video\n";
        return ;
    }

    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);  //Set width to 1920 (1080p)
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080); //Set height to 1080

   
    Mat frame;
    int skip_frames = 2; // skip every 2 frames
    int frame_count = 0;

    while (cap.read(frame)) {
        frame_count++;
        if (frame_count % skip_frames != 0) continue; // skip frame
        Mat blob;
        blobFromImage(frame, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);     //Preprocess the frame (resize, normalize, create blob)
        net.setInput(blob);

        //Forward pass (run inference)
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        postprocess(frame, outs, net, classes);
        imshow("YOLOv3 Detection (Webcam)", frame);
        if (waitKey(1) == 'q') break;
    }


    cap.release();
}

void detectVideo(const string &video, Net &net, vector<string> &classes)
{
    VideoCapture cap(video);

    //Get Video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int codec = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

    std::cout << "Video Properties:\n"
    << "Resolution: " << width << "x" << height << "\n"
    << "FPS: " << fps << std::endl;
    Mat frame;
    int skip_frames = 2; // skip every 2 frames
    int frame_count = 0;

     while (cap.read(frame)) {
        frame_count++;
        if (frame_count % skip_frames != 0) continue; // skip frame
        Mat blob;
        blobFromImage(frame, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        postprocess(frame, outs, net, classes);
        imshow("YOLOv3 Detection (Video)", frame);
        if (waitKey(1) == 'q') break;
    }


    cap.release();
}

// Setters
void setImage(string image)
{
    this->imagePath = image;
}

void setVideo(string video)
{
    this->videoPath = video;
}

void setCam(int webcam)
{
    this->cam = webcam;
}

// Getters
string getImage()
{
    return this->imagePath;
}

string getVideo()
{
    return this->videoPath;
}

int getCam()
{
    return this->cam;
}


private:
 string imagePath;
 string videoPath;
 int cam;

};

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);  //silence logs

    string classesFile = "/yolo/coco.names";
    string modelConfiguration = "/yolo/yolov3.cfg";
    string modelWeights = "/yolo/yolov3.weights";

   

    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Type image, video, cam;
    image.setImage("/Hisham/YOLO_test/YOLO_practice/img.jpg");
    image.detectImage(image.getImage(),net,classes);

    video.setVideo("/Hisham/YOLO_test/YOLO_practice/object_detection_test.mp4");
    video.detectVideo(video.getVideo(),net,classes);
    
    cam.setCam(0);
    cam.detectWebcam(cam.getCam(),net,classes);
    
    destroyAllWindows();
    return 0;
}
