# OpenCV_CPP_Project

**Author:** [Hisham Elsayed](https://github.com/Hisham-Elsayed)

---

## Overview

OpenCV_CPP_Project is a modern C++ application demonstrating object detection using OpenCV and the YOLO (You Only Look Once) algorithm.  
The project supports detection on images, videos, and live camera feeds, using a modular, object-oriented design.

---

## Features

- **Image Detection:** Detect objects in static images.
- **Video Detection:** Detect objects frame-by-frame in video files.
- **Real-Time Camera Detection:** Detect objects from a live camera feed.
- **Multiple YOLO Models:** Easily use different YOLO models (e.g., YOLOv4, YOLOv4-tiny) in the same run.
- **OOP Design:** Encapsulated YOLO logic, dependency injection, and polymorphic detectors.
- **Extensible:** Add new detectors or models with minimal code changes.

---

## Project Structure

```
.
├── CMakeLists.txt
├── cameraDetection.cpp/h
├── imageDetection.cpp/h
├── main.cpp
├── type.h
├── videoDetection.cpp/h
├── yolo.cpp/h
├── Samples/
│   ├── Images/
│   │   ├── dog_bike_car.jpg
│   │   └── img.jpg
│   └── Videos/
│       ├── object_detection_test.mp4
│       └── Vehicle Dataset Sample 2.mp4
```

---

## Architecture

### Class Diagram (PlantUML)

```plantuml
@startuml
abstract class Type {
    - windowName : std::string
    + detect() : void
    + ~Type()
}

class YoloDetector {
    - classesFile : std::string
    - modelConfiguration : std::string
    - modelWeights : std::string
    - net : cv::dnn::Net
    - classes : std::vector<std::string>
    + YoloDetector(class_file, config_file, weights_file)
    + load() : bool
    + getNet() : cv::dnn::Net&
    + getOutputsNames() : std::vector<std::string>
    + drawPred(classId, conf, left, top, right, bottom, frame) : void
    + postprocess(frame, outs) : void
    + confThreshold : static constexpr float
    + nmsThreshold : static constexpr float
    + inpWidth : static constexpr int
    + inpHeight : static constexpr int
}

class ImageDetector {
    - imagePath : std::string
    - yolo : YoloDetector&
    + ImageDetector(path, yolo)
    + detect() : void
}

class VideoDetector {
    - videoPath : std::string
    - yolo : YoloDetector&
    + VideoDetector(path, yolo)
    + detect() : void
}

class CameraDetector {
    - cam : int
    - yolo : YoloDetector&
    + CameraDetector(camIndex, yolo)
    + detect() : void
}

Type <|-- ImageDetector
Type <|-- VideoDetector
Type <|-- CameraDetector

ImageDetector o-- YoloDetector : uses
VideoDetector o-- YoloDetector : uses
CameraDetector o-- YoloDetector : uses
@enduml
```

### Sequence Diagram (PlantUML)

```plantuml
@startuml
actor User
participant Main as "main()"
participant ImageDetector
participant VideoDetector
participant CameraDetector
participant YoloDetector
participant OpenCV as "cv::dnn::Net"

User -> Main : start()
Main -> YoloDetector : load() (for each model)
YoloDetector -> YoloDetector : load class names\nload network weights

Main -> ImageDetector : create ImageDetector(imagePath, yolo)
Main -> VideoDetector : create VideoDetector(videoPath, yolo)
Main -> CameraDetector : create CameraDetector(camIndex, yolo)

Main -> ImageDetector : detect()
ImageDetector -> ImageDetector : load image
ImageDetector -> YoloDetector : getNet()
ImageDetector -> YoloDetector : getOutputsNames()
ImageDetector -> OpenCV : setInput(blob)
ImageDetector -> OpenCV : forward(outs, outputNames)
ImageDetector -> YoloDetector : postprocess(image, outs)
YoloDetector -> YoloDetector : drawPred(..., image)
ImageDetector -> ImageDetector : show image

Main -> VideoDetector : detect()
VideoDetector -> VideoDetector : open video file
loop for each frame
    VideoDetector -> YoloDetector : getNet()
    VideoDetector -> YoloDetector : getOutputsNames()
    VideoDetector -> OpenCV : setInput(blob)
    VideoDetector -> OpenCV : forward(outs, outputNames)
    VideoDetector -> YoloDetector : postprocess(frame, outs)
    YoloDetector -> YoloDetector : drawPred(..., frame)
    VideoDetector -> VideoDetector : show frame
end
VideoDetector -> VideoDetector : release video

Main -> CameraDetector : detect()
CameraDetector -> CameraDetector : open camera
loop for each frame
    CameraDetector -> YoloDetector : getNet()
    CameraDetector -> YoloDetector : getOutputsNames()
    CameraDetector -> OpenCV : setInput(blob)
    CameraDetector -> OpenCV : forward(outs, outputNames)
    CameraDetector -> YoloDetector : postprocess(frame, outs)
    YoloDetector -> YoloDetector : drawPred(..., frame)
    CameraDetector -> CameraDetector : show frame
end
CameraDetector -> CameraDetector : release camera

@enduml
```

---

## Prerequisites

- **Operating System:** Windows, Linux, or macOS
- **Compiler:** C++17 compatible (MSVC, GCC, Clang)
- **CMake:** Version 3.10 or higher
- **OpenCV:** Version 4.x installed and configured
- **YOLO Files:** Place `coco.names`, `yolov4-tiny.cfg`, `yolov4-tiny.weights`, etc. in `C:\yolo\` (or update the paths in `main.cpp`)

---

## Building the Project

### **On Windows (Visual Studio 2022)**

```sh
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build .
```

- To run the app (Debug build):
  ```
  ./Debug/opencv_cpp.exe
  ```
  Or (Release build):
  ```
  ./Release/opencv_cpp.exe
  ```

### **On Linux/macOS**

```sh
mkdir build
cd build
cmake ..
make
./opencv_cpp
```

---

## Usage

- The application will process:
  - Images from `Samples/Images/`
  - Videos from `Samples/Videos/`
  - Camera feed (if available)
- You can add or change detectors in `main.cpp` as needed.

---

## YOLO Integration

- The project uses the YOLO algorithm for object detection.
- Ensure the required YOLO config and weights files are present and the paths in `main.cpp` are correct.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## Acknowledgments

- [OpenCV](https://github.com/opencv/opencv)
- [YOLOv4](https://github.com/AlexeyAB/darknet)
- The creators of the YOLO algorithm for their groundbreaking work in real-time object detection.

---