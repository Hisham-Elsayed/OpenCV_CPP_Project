#include "cameraDetection.h"
#include "yolo.h"

using namespace cv;
using namespace dnn;
using namespace std;

/**
 * @brief Constructs a CameraDetector with the given camera index.
 * @param path Camera index (usually 0 for default camera).
 */
CameraDetector::CameraDetector(const int &path) : cam(path){}

/**
 * @brief Detects objects from the camera feed and displays the result in real time.
 * @param net Reference to the loaded YOLO network.
 * @param classes Vector of class names.
 */
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

    windowName = "YOLOv4-tiny Camera Detection";
    
    // Display the camera with resizable window
    namedWindow(windowName, WINDOW_NORMAL); 

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

        imshow(windowName, frame);
        if (waitKey(1) == 'q') break;
    }


    cap.release();
}