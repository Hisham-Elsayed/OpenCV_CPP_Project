#include "imageDetection.h"


using namespace cv;
using namespace dnn;
using namespace std;

/**
 * @brief Constructs an ImageDetector with the given image path.
 * @param path Path to the image file.
 * @param yolo Reference to a YoloDetector instance to use for detection.
 */
ImageDetector::ImageDetector(const std::string &path, YoloDetector& yolo) : imagePath(path), yolo(yolo) {}

/**
 * @brief Detects objects in the image and displays the result.
 */
void ImageDetector::detect()
{
    Mat image = imread(imagePath);

    if (image.empty()) {
        cerr << "Failed to load image from " << imagePath << endl;
        return;
    }

    // Create a 4D blob from the image
    Mat blob;
    blobFromImage(image, blob, 1/255.0, Size(YoloDetector::inpWidth, YoloDetector::inpHeight), Scalar(0,0,0), true, false);

    // Set the input to the network
    yolo.getNet().setInput(blob);

    // Forward pass
    vector<Mat> outs;
    yolo.getNet().forward(outs, yolo.getOutputsNames());

    // Process detections and draw bounding boxes
    yolo.postprocess(image, outs);

    windowName = "Image Detection";
    
    // Display the image with resizable window
    namedWindow(windowName, WINDOW_NORMAL); 

    // Display the image
    imshow(windowName, image);
    waitKey(0); // Wait for any key press to close the window
}