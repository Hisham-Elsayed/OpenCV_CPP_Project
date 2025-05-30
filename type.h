#ifndef TYPE_H
#define TYPE_H


#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace dnn;
using namespace std;

class Type
{
    public:
    virtual void detect(Net& net, vector<string>& classes) = 0; //pure virtual function

    virtual ~Type() = default; // virtual destructor
};

#endif