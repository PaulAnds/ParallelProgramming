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