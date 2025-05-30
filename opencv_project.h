#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

class Type
{
public:
void detectPhoto(const string& imagePath, Net& net, vector<string>& classes);
void detectWebcam(const int &cam, Net &net, vector<string> &classes);   
void detectVideo(const string &video, Net &net, vector<string> &classes);

private:
 string imagePath;
 string videoPath;

};