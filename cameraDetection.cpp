#include "cameraDetection.h"
#include "yolo.h"

using namespace cv;
using namespace dnn;
using namespace std;

CameraDetector::CameraDetector(const int &path) : cam(path){}

void CameraDetector::detect(cv::dnn::Net &net, std::vector<std::string>& classes)
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