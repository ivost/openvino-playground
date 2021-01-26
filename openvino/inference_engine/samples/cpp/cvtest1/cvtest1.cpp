/*
* 
 
    https://docs.opencv.org/4.5.1/
 
    https://www.murtazahassan.com/courses/opencv-cpp-course/lesson/windows/

    VC++ Directories
    1. Add Build Directories: D:\opencv\build\include
    2. Add Library Directories: D:\opencv\build\x64\vc15\lib
Linker Input
    3. Add Linker input: opencv_world451d.lib (opencv 4.5.1)
       d for debug without d for release

*/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;


void showImage(string path) {
    Mat img = imread(path);
    imshow("Image", img);
}

void showVideo(string path) {
    VideoCapture vc(path);
    Mat m;
    while (vc.read(m)) {
        imshow("Image", m);
        int k = waitKey(20);
        if (k >= ' ') {
            return;
        }
    }
}

void showCam(int n) {
    VideoCapture vc(n);
    Mat m;
    while (vc.read(m)) {
        imshow("Image", m);
        int k = waitKey(1);
        if (k >= ' ') {
            return;
        }
    }
}

void effects(string path) {
    Mat img = imread(path);
    Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(7, 7), 5, 0);
    Canny(imgBlur, imgCanny, 25, 75);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCanny, imgDil, kernel);
    erode(imgDil, imgErode, kernel);

    imshow("Image", img);
    imshow("Image Gray", imgGray);
    imshow("Image Blur", imgBlur);
    imshow("Image Canny", imgCanny);
    imshow("Image Dilation", imgDil);
    imshow("Image Erode", imgErode);
    waitKey(0);
}

void resizeAndCrop(string path) {
    Mat img = imread(path);
    Mat imgResize, imgCrop;

    //cout << img.size() << endl;
    resize(img, imgResize, Size(), 0.5, 0.5);

    Rect roi(200, 100, 300, 300);
    imgCrop = img(roi);

    imshow("Image", img);
    imshow("Image Resize", imgResize);
    imshow("Image Crop", imgCrop);
    waitKey(0);
}

void shapesAndText() {
    // Blank Image
    Mat img(512, 512, CV_8UC3, Scalar(255, 255, 255));

    circle(img, Point(256, 256), 155, Scalar(0, 69, 255), FILLED);
    rectangle(img, Point(130, 226), Point(382, 286), Scalar(255, 255, 255), FILLED);
    line(img, Point(130, 296), Point(382, 296), Scalar(255, 255, 255), 2);

    putText(img, "Some text", Point(137, 262), FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 69, 255), 2);

    imshow("Image", img);
}

void colorDetection(string path) {
    Mat img = imread(path);
    Mat imgHSV, mask;
    int hmin = 0, smin = 110, vmin = 153;
    int hmax = 19, smax = 240, vmax = 255;

    cvtColor(img, imgHSV, COLOR_BGR2HSV);

    namedWindow("Trackbars", (640, 200));
    createTrackbar("Hue Min", "Trackbars", &hmin, 179);
    createTrackbar("Hue Max", "Trackbars", &hmax, 179);
    createTrackbar("Sat Min", "Trackbars", &smin, 255);
    createTrackbar("Sat Max", "Trackbars", &smax, 255);
    createTrackbar("Val Min", "Trackbars", &vmin, 255);
    createTrackbar("Val Max", "Trackbars", &vmax, 255);

    while (true) {

        Scalar lower(hmin, smin, vmin);
        Scalar upper(hmax, smax, vmax);
        inRange(imgHSV, lower, upper, mask);

        imshow("Image", img);
        imshow("Image HSV", imgHSV);
        imshow("Image Mask", mask);
        waitKey(1);
    }
}


void getContours(Mat imgDil, Mat img) {

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());

    for (int i = 0; i < contours.size(); i++)
    {
        int area = contourArea(contours[i]);
        cout << area << endl;
        string objectType;

        if (area > 1000)
        {
            float peri = arcLength(contours[i], true);
            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
            cout << conPoly[i].size() << endl;
            boundRect[i] = boundingRect(conPoly[i]);

            int numVert = (int)conPoly[i].size();

            if (numVert == 3) { objectType = "Triangle"; }
            else if (numVert == 4)
            {
                float aspRatio = (float)boundRect[i].width / (float)boundRect[i].height;
                cout << aspRatio << endl;
                if (aspRatio > 0.95 && aspRatio < 1.05) { objectType = "Square"; }
                else { objectType = "Rectangle"; }
            }
            else if (numVert > 4) { objectType = "Circle"; }

            drawContours(img, conPoly, i, Scalar(255, 0, 255), 2);
            rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
            putText(img, objectType, { boundRect[i].x,boundRect[i].y - 5 }, FONT_HERSHEY_PLAIN, 1, Scalar(0, 69, 255), 2);
        }
    }
}

    void face(string path) {
        Mat img = imread(path);

        CascadeClassifier faceCascade;
        faceCascade.load("Resources/haarcascade_frontalface_default.xml");

        if (faceCascade.empty()) { 
            cout << "XML file not loaded" << endl; 
            return;
        }

        vector<Rect> faces;
        faceCascade.detectMultiScale(img, faces, 1.1, 10);

        for (int i = 0; i < faces.size(); i++)
        {
            rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
        }

        imshow("Image", img);
        waitKey(0);
    }


int main() {

    /*
    showImage("Resources/lambo.png");
    showVideo("Resources/test_video.mp4");
    showCam(0);

    effects("Resources/lambo.png");
    resizeAndCrop("Resources/lambo.png");
    shapesAndText();
    colorDetection("Resources/lambo.png");

    string path = "Resources/shapes.png";
    Mat img = imread(path);
    Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

    // Preprocessing
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
    Canny(imgBlur, imgCanny, 25, 75);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCanny, imgDil, kernel);

    getContours(imgDil, img);

    imshow("Image", img);
    imshow("Image Dilated", imgDil);


    */

    string path = "Resources/test.png";
    face(path);

    waitKey(0);
    return 0;
}



