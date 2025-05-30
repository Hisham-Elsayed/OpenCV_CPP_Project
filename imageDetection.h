#ifndef IMAGE_H
#define IMAGE_H

#include "type.h"

class ImageDetector : public Type
{
    public:
    ImageDetector(const std::string &path);
    void detect(cv::dnn::Net& net, std::vector<std::string>& classes) override;


    private:
    std::string imagePath;
};





#endif