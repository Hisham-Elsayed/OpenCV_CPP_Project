#include "cameraDetection.h"


using namespace cv;
using namespace dnn;
using namespace std;

/**
 * @brief Constructs a CameraDetector with the given camera index.
 * @param camIndex Camera index (usually 0 for default camera).
 * @param yolo Reference to a YoloDetector instance to use for detection.
 */
CameraDetector::CameraDetector(const int &camIndex, YoloDetector& yolo) : cam(camIndex), yolo(yolo){}

/**
 * @brief Detects objects from the camera feed and displays the result in real time.
 */
void CameraDetector::detect()
{
    VideoCapture cap(cam);        //to use webcam
    if (!cap.isOpened()) {
        cerr << "Error opening video\n";
        return ;
    }

    Mat frame;
    int skip_frames = 2; // skip every 2 frames
    int frame_count = 0;

    windowName = "Camera Detection";
    
    // Display the camera with resizable window
    namedWindow(windowName, WINDOW_NORMAL); 

    while (cap.read(frame)) {
        frame_count++;
        if (frame_count % skip_frames != 0) continue; // skip frame
        Mat blob;
        blobFromImage(frame, blob, 1/255.0, Size(YoloDetector::inpWidth, YoloDetector::inpHeight), Scalar(0,0,0), true, false);     //Preprocess the frame (resize, normalize, create blob)
        
        yolo.getNet().setInput(blob);

        //Forward pass (run inference)
        vector<Mat> outs;
        yolo.getNet().forward(outs, yolo.getOutputsNames());

        yolo.postprocess(frame, outs);

        imshow(windowName, frame);
        if (waitKey(1) == 'q') break;
    }


    cap.release();
}