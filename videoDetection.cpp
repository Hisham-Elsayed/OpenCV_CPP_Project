#include "videoDetection.h"


using namespace cv;
using namespace dnn;
using namespace std;

/**
 * @brief Constructs a VideoDetector with the given video path.
 * @param path Path to the video file.
 * @param yolo Reference to a YoloDetector instance to use for detection.
 */
VideoDetector::VideoDetector(const std::string &path, YoloDetector& yolo) : videoPath(path), yolo(yolo){}

/**
 * @brief Detects objects in the video and displays the result frame by frame.
 */
void VideoDetector::detect()
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

    windowName = "Video Detection";

    // Display the video with resizable window
    namedWindow(windowName, WINDOW_NORMAL); 

    while (cap.read(frame)) {
        frame_count++;
        if (frame_count % skip_frames != 0) continue; // skip frame
        Mat blob;
        blobFromImage(frame, blob, 1/255.0, Size(YoloDetector::inpWidth, YoloDetector::inpHeight), Scalar(0,0,0), true, false);
        yolo.getNet().setInput(blob);
        vector<Mat> outs;
        yolo.getNet().forward(outs, yolo.getOutputsNames());

        yolo.postprocess(frame, outs);

        imshow(windowName, frame);
        if (waitKey(1) == 'q') break;
    }
}