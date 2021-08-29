// ImageStitching.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
int main()
{
    
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    std::vector<cv::KeyPoint> lastFramekeypoints1, lastFramekeypoints2;
    cv::Mat lastFrameDescriptors1, lastFrameDescriptors2;
    std::vector<cv::DMatch > matches, resultmatches;
    int movementDirection = 0;
    
    std::string image_path1 = samples::findFile("E:\\Pavan\\VisualStudioCodes\\ImageStitching\\Images\\Hill1.jpg");
    std::string image_path2 = samples::findFile("E:\\Pavan\\VisualStudioCodes\\ImageStitching\\Images\\Hill2.jpg");
    Mat image1 = imread(image_path1);
    Mat image2 = imread(image_path2);

    // finds keypoints and their disriptors
    detector->detectAndCompute(image1, noArray(), lastFramekeypoints1, lastFrameDescriptors1);
    detector->detectAndCompute(image2, noArray(), lastFramekeypoints2, lastFrameDescriptors2);

    // match the descriptor between two images
    matcher->match(lastFrameDescriptors1, lastFrameDescriptors2, matches);

    std::vector<cv::Point2d> good_point1, good_point2;
    good_point1.reserve(matches.size());
    good_point2.reserve(matches.size());

    //calculation of max and min distances between keypoints
    double max_dist = 0; double min_dist = 100;
    for (const auto& m : matches)
    {
        double dist = m.distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    // filter out good points
    // distance which is less than or equals the min_dist*1.5
    // if this value is increased more keypoits are detected.
    for (const auto& m : matches)
    {
        if (m.distance <= 1.5 * min_dist)
        {
            // the matches variable holds the index values of x-y positions of the keypoints in both images.
            // queryIdx gives key points index which has a match and trainIdx gives its corrosponding matched key point position index 
            // these inde values then can be used to find the key points in the key points arrays.
            // e.g. array1[(1,20),(3,40)] array2[(5,40),(6,20)] 
            // m.queryIdx=0 m.trainIdx=1 => array1[0] array2[1] has a match. array2 co-ordinates are found at array2 given position.

            good_point1.push_back(lastFramekeypoints1.at(m.queryIdx).pt);
            good_point2.push_back(lastFramekeypoints2.at(m.trainIdx).pt);
        }
    }

    // crop rectangle constructor.
    cv::Rect croppImg1(0, 0, image1.cols, image1.rows);
    cv::Rect croppImg2(0, 0, image2.cols, image2.rows); 

    // find minimum horizontal value for image 1 to crop
    //e.g. img1 size = 200 first keypoint having match found at position 100 crop img1 to 0-100
    // crop image2 to from corresponding x value to the width. 
    //e.g. img2 width 200 point found at 50 crop image  50-200

    // movementDirection tells us are both the images aligned or not if not adjust the images accordingly.
    int imgWidth = image1.cols;
    for (int i = 0; i < good_point1.size(); ++i)
    {
        if (good_point1[i].x < imgWidth)
        {
            croppImg1.width = good_point1.at(i).x;
            croppImg2.x = good_point2[i].x;
            croppImg2.width = image2.cols - croppImg2.x;
            movementDirection = good_point1[i].y - good_point2[i].y;
            imgWidth = good_point1[i].x;
        }
    }
    image1 = image1(croppImg1);
    image2 = image2(croppImg2);
    int maxHeight = image1.rows > image2.rows ? image1.rows : image2.rows;
    int maxWidth = image1.cols + image2.cols;
    cv::Mat result=cv::Mat::zeros(cv::Size(maxWidth, maxHeight + abs(movementDirection)), CV_8UC3);
    if (movementDirection > 0)
    {
        cv::Mat half1(result, cv::Rect(0, 0, image1.cols, image1.rows));
        image1.copyTo(half1);
        cv::Mat half2(result, cv::Rect(image1.cols, abs(movementDirection),image2.cols, image2.rows));
        image2.copyTo(half2);
    }
    else
    {
        cv::Mat half1(result, cv::Rect(0, abs(movementDirection), image1.cols, image1.rows));
        image1.copyTo(half1);
        cv::Mat half2(result, cv::Rect(image1.cols,0 ,image2.cols, image2.rows));
        image2.copyTo(half2);
    }
    imshow("Stitched Image", result);
    
    int k = waitKey(0); // Wait for a keystroke in the window
    if (k == 's')
    {
        imwrite("StitchedImage.png", result);
    }
    return 0;
}

