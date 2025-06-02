#include "videoDetection.h"
#include "yolo.h"

using namespace cv;
using namespace dnn;
using namespace std;

VideoDetector::VideoDetector(const std::string &path) : videoPath(path){}

void VideoDetector::detect(cv::dnn::Net &net, std::vector<std::string>& classes)
{
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error opening video\n";
        return ;
    }

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

        windowName = "YOLOv4-tiny video Detection";

        // Display the image with resizable window
        namedWindow(windowName, WINDOW_NORMAL); 

        imshow(windowName, frame);
        if (waitKey(1) == 'q') break;
    }
}