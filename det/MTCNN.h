//
// Created by Young on 2016/11/27.
//

//#define CPU_ONLY

#ifndef MTCNN_MTCNN_H
#define MTCNN_MTCNN_H

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace caffe;

class MTCNN {
 public:
  MTCNN();
  MTCNN(const std::vector<std::string> model_file,
        const std::vector<std::string> trained_file);
  ~MTCNN();
  void PNetRun(const cv::Mat& image, std::vector<float>& confidences,
               std::vector<cv::Rect>& bboxes);
  void RNetRun(const cv::Mat& image, std::vector<cv::Rect>& pro_bboxes,
               std::vector<float>& confidences, std::vector<cv::Rect>& bboxes);

  void ONetRun(const cv::Mat& image, std::vector<cv::Rect>& p_bboxes,
               std::vector<float>& confidences, std::vector<cv::Rect>& bboxes);

  void NMS(std::vector<cv::Rect>& cur_rects, std::vector<float>& confidence,
           float threshold);

  cv::Mat crop(cv::Mat img, cv::Rect& rect);
  float IoU(cv::Rect rect1, cv::Rect rect2);
  float IoM(cv::Rect rect1, cv::Rect rect2);
  void global_NMS();

  std::shared_ptr<caffe::Net> pnet_;
  std::shared_ptr<caffe::Net> rnet_;
  std::shared_ptr<caffe::Net> onet_;
  cv::Size pnet_input_size_;
  cv::Size rnet_input_size_;
  cv::Size onet_input_size_;
  float pnet_thresh_ = 0.6;
  float rnet_thresh_ = 0.2;
  float onet_thresh_ = 0.7;

  // paramter for the threshold
  int minSize_ = 40;
  float factor_ = 0.709;
  float threshold_[3] = {0.6, 0.2, 0.7};
  float threshold_NMS_ = 0.4;
  std::vector<std::vector<string>> output_blob_names_ = {
      {"conv4-2", "prob1"},
      {"conv5-2", "prob1"},
      {"conv6-2", "conv6-1", "prob1"}};
};

#endif  // MTCNN_MTCNN_H
