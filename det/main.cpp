#include <time.h>
#include <iostream>
#include "MTCNN.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  vector<string> model_file = {"../model/det1.prototxt",
                               "../model/det2.prototxt",
                               "../model/det3.prototxt"

  };

  vector<string> trained_file = {"../model/det1.caffemodel",
                                 "../model/det2.caffemodel",
                                 "../model/det3.caffemodel"

  };

  MTCNN mtcnn(model_file, trained_file);
  int count = 1;
  unsigned start = time(NULL);
  vector<Rect> rectangles;
  string img_path = argv[1];
  Mat img = imread(img_path);
  std::vector<float> confs;
  std::vector<cv::Rect> bboxes;
  mtcnn.PNetRun(img, confs, bboxes);
  std::vector<float> rconfs;
  std::vector<cv::Rect> rbboxes;
  mtcnn.RNetRun(img, bboxes, rconfs, rbboxes);
  std::vector<float> oconfs;
  std::vector<cv::Rect> obboxes;
  mtcnn.ONetRun(img, rbboxes, oconfs, obboxes);
  std::cout << "h w " << img.rows << " " << img.cols << std::endl;

  unsigned end = time(NULL);
  unsigned ave = (end - start) * 1000.0 / count;
  std::cout << "Run " << count << " times, "
            << "Average time:" << ave << std::endl;

  return 0;
}
