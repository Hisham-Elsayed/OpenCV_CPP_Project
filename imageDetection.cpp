#include "imageDetection.h"
#include "yolo.h"

using namespace cv;
using namespace dnn;
using namespace std;

/**
 * @brief Constructs an ImageDetector with the given image path.
 * @param path Path to the image file.
 */
ImageDetector::ImageDetector(const std::string &path) : imagePath(path){}

/**
 * @brief Detects objects in the image and displays the result.
 * @param net Reference to the loaded YOLO network.
 * @param classes Vector of class names.
 */
void ImageDetector::detect(cv::dnn::Net& net, const std::vector<std::string>& classes)
{
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

    windowName = "YOLOv4-tiny Image Detection";
    
    // Display the image with resizable window
    namedWindow(windowName, WINDOW_NORMAL); 

    // Display the image
    imshow(windowName, image);
    waitKey(0); // Wait for any key press to close the window
}