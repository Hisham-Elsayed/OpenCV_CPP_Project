#ifndef VIDEO_DETECTION_H
#define VIDEO_DETECTION_H

#include "type.h"

class VideoDetector : public Type
{
    public:
    VideoDetector(const string &path);
    void detect(Net &net, vector<std::string>& classes) override;

    private:
    string videoPath;
};


#endif