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

        windowName = "YOLOv4-tiny Camera Detection";

        // Display the image with resizable window
        namedWindow(windowName, WINDOW_NORMAL); 

        imshow(windowName, frame);
        if (waitKey(1) == 'q') break;
    }


    cap.release();
}