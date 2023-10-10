# OpenCV Image Processing C++ Code

This is a simple C++ program that uses the OpenCV library to perform basic image processing operations on an input image. The program accomplishes the following tasks:

<img src="https://imgs.search.brave.com/QokKV8YJWVzjFxgihzHxLfZKNK7zCA2TgND_5I35SCU/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9sZWFy/bm9wZW5jdi5jb20v/d3AtY29udGVudC91/cGxvYWRzLzIwMTUv/MDcvY29sb3JtYXBf/b3BlbmN2X2V4YW1w/bGUuanBn" width="500" height="300" />

### Image Loading: 
It prompts the user to enter the path to an image file. If the specified file exists, it loads the image using OpenCV.

### Image Display: 
If the image is successfully loaded, it displays the original image in a window titled "Image."

### Color Channel Separation: 
The program separates the loaded image into its three color channels: blue, green, and red.

### Individual display of Gray intensity 
The program shows the intensity of each indivudal channel on its respective grayscale.

### Color Mapping: 
It applies different color maps to each of the separated color channels. Specifically, it uses the following color maps:

# Code

```
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <filesystem>

int main() {
    std::string image_path;
    std::cout << "Enter the path to the image: ";
    std::cin >> image_path;

    if (!std::filesystem::exists(image_path)) {
        std::cout << "File does not exist at the specified path" << std::endl;
        return -1;
    }

    cv::Mat originalImage = cv::imread(image_path);

    if (originalImage.empty()) {
        std::cout << "Error loading the image" << std::endl;
        return -1;
    }
    else {
        std::cout << "Image loaded successfully" << std::endl;
    }

    cv::imshow("Image", originalImage);

    cv::waitKey(0);
    cv::destroyAllWindows();

    // Separate the image into its three channels
    cv::Mat bgr[3];
    cv::split(originalImage, bgr);

    // Show individual channels with their gray intensity
    cv::imshow("grayBlueChannel", bgr[0]);
    cv::imshow("grayGreenChannel", bgr[1]);
    cv::imshow("grayRedChannel", bgr[2]);

    //Wait for a keystroke in the window
    cv::waitKey(0);
    cv::destroyAllWindows();

    //Modify images by color maps
    cv::Mat blueChannel, greenChannel, redChannel;
    cv::applyColorMap(bgr[0], blueChannel, cv::COLORMAP_BONE);
    cv::applyColorMap(bgr[1], greenChannel, cv::COLORMAP_COOL);
    cv::applyColorMap(bgr[2], redChannel, cv::COLORMAP_HSV);

    // Create modified image windows
    cv::imshow("blueChannel", blueChannel);
    cv::imshow("greenChannel", greenChannel);
    cv::imshow("redChannel", redChannel);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
```

To split the image i use an array of mat's called brg[] and use the opencv code split to split it into 3 images in the bgr array
then show the splitted images to show their individual intensities.

#### For the color maps i used pre-defined maps in opencv 

1. Blue Channel: COLORMAP_BONE

2. Green Channel: COLORMAP_COOL

3. Red Channel: COLORMAP_HSV

User Interaction: To view the modified channels, the user needs to press any key in the respective channel windows.

## References:

Wagner, P. (2011) Colormaps in opencv, httpswwwbytefishde ATOM. Available at: https://www.bytefish.de/blog/colormaps_in_opencv.html (Accessed: 09 October 2023). 

Yip, M. (2020) OpenCV: Detect whether a window is closed. or close by press ‘x’ button., Medium. Available at: https://medium.com/@mh_yip/opencv-detect-whether-a-window-is-closed-or-close-by-press-x-button-ee51616f7088 (Accessed: 09 October 2023). 