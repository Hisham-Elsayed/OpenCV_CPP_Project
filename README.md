
# OpenCV_CPP_Project

**Author:** [Hisham Elsayed](https://github.com/Hisham-Elsayed)

## Overview

OpenCV_CPP_Project is a C++ application that demonstrates object detection capabilities using OpenCV and the YOLO (You Only Look Once) algorithm. The project supports processing images, videos, and real-time camera feeds to detect and annotate objects within them.

## Features

- **Image Detection:** Process static images to detect objects.
- **Video Detection:** Analyze video files frame by frame for object detection.
- **Real-Time Camera Detection:** Utilize a connected camera to perform live object detection.
- **Modular Design:** Organized codebase with separate modules for image, video, and camera processing.
- **YOLO Integration:** Implements the YOLO algorithm for efficient and accurate object detection.

## Project Structure

```
.
├── .vscode/                   # Visual Studio Code configuration files
├── CMakeLists.txt             # CMake build configuration
├── cameraDetection.cpp        # Implementation for camera-based detection
├── cameraDetection.h          # Header for camera detection
├── imageDetection.cpp         # Implementation for image-based detection
├── imageDetection.h           # Header for image detection
├── img.jpg                    # Sample image for testing
├── dog_bike_car.jpg                    # Sample image for testing
├── main.cpp                   # Entry point of the application
├── object_detection_test.mp4  # Sample video for testing
├── type.h                     # Definitions of custom data types
├── videoDetection.cpp         # Implementation for video-based detection
├── videoDetection.h           # Header for video detection
├── yolo.cpp                   # YOLO algorithm implementation
└── yolo.h                     # Header for YOLO implementation
```

## Prerequisites

- **Operating System:** Compatible with Windows, macOS, and Linux.
- **Compiler:** C++17 compatible compiler (e.g., GCC, Clang, MSVC).
- **CMake:** Version 3.10 or higher.
- **OpenCV:** Version 4.x installed and configured.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Hisham-Elsayed/OpenCV_CPP_Project.git
   cd OpenCV_CPP_Project
   ```

2. **Build the Project:**

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

   Ensure that OpenCV is properly installed and its paths are correctly set in your environment.

## Usage

After building the project, you can execute the application with different modes:

```bash
./OpenCV_CPP_Project [mode] [input_path]
```

- `[mode]`: Specify the mode of operation:
  - `image`: Process a static image.
  - `video`: Process a video file.
  - `camera`: Use the connected camera for real-time detection.

- `[input_path]`: Path to the input file (image or video). Not required for `camera` mode.

### Examples

- **Image Detection:**

  ```bash
  ./OpenCV_CPP_Project image ../img.jpg
  ```

- **Video Detection:**

  ```bash
  ./OpenCV_CPP_Project video ../object_detection_test.mp4
  ```

- **Camera Detection:**

  ```bash
  ./OpenCV_CPP_Project camera
  ```

## YOLO Integration

The project integrates the YOLO algorithm for object detection. Ensure that the necessary YOLO configuration files and weights are available and correctly referenced in the `yolo.cpp` and `yolo.h` files.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://github.com/opencv/opencv) for providing the computer vision library.
- The creators of the YOLO algorithm for their groundbreaking work in real-time object detection.
