#include "imageDetection.h"
#include "yolo.h"

using namespace cv;
using namespace dnn;
using namespace std;

ImageDetector::ImageDetector(const std::string &path) : imagePath(path){}

void ImageDetector::detect(cv::dnn::Net& net, std::vector<std::string>& classes)
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

    // Display the image
    imshow("YOLOv3 Image Detection", image);
    waitKey(0); // Wait for any key press to close the window
}