//
// Created by Young on 2016/11/27.
//

#include "MTCNN.h"
MTCNN::MTCNN() {}

MTCNN::MTCNN(const std::vector<std::string> model_file,
             const std::vector<std::string> trained_file) {
#ifdef CPU_ONLY
  caffe::SetMode(caffe::CPU, -1);
  std::cout << '\n' << "USE CPU" << '\n';  // by yzh
#else
  caffe::SetMode(caffe::GPU, 0);
  std::cout << '\n' << "USE GPU" << '\n';  // by yzh
#endif

  pnet_.reset(new caffe::Net(model_file[0]));
  pnet_->CopyTrainedLayersFrom(trained_file[0]);
  caffe::Blob* pnet_input = pnet_->blob_by_name("data").get();
  pnet_input_size_ = cv::Size(pnet_input->width(), pnet_input->height());

  rnet_.reset(new caffe::Net(model_file[1]));
  rnet_->CopyTrainedLayersFrom(trained_file[1]);
  caffe::Blob* rnet_input = rnet_->blob_by_name("data").get();
  rnet_input_size_ = cv::Size(rnet_input->width(), rnet_input->height());

  onet_.reset(new caffe::Net(model_file[2]));
  onet_->CopyTrainedLayersFrom(trained_file[2]);
  caffe::Blob* onet_input = onet_->blob_by_name("data").get();
  onet_input_size_ = cv::Size(onet_input->width(), onet_input->height());
}

MTCNN::~MTCNN() {}

void MTCNN::PNetRun(const cv::Mat& image, std::vector<float>& confidences,
                    std::vector<cv::Rect>& bboxes) {
  int msize = (image.rows < image.cols) ? image.rows : image.cols;
  int raw_image_row = image.rows;
  int raw_image_col = image.cols;
  int cur_size = 12;
  float m = float(cur_size) / minSize_;
  std::vector<float> scales;
  while (msize * m > cur_size) {
    scales.push_back(m);
    m *= factor_;
  }

  for (int i = 0; i < scales.size(); ++i) {
    int new_h = (int)std::ceil(image.rows * scales[i]);
    int new_w = (int)std::ceil(image.cols * scales[i]);
    cv::Mat retImage;
    cv::resize(image, retImage, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::imwrite("ss_" + std::to_string(scales[i]) + ".jpg", retImage);
    // Preprocess(retImage);
    retImage.convertTo(retImage, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
    caffe::Blob* input_layer = pnet_->blob_by_name("data").get();
    input_layer->Reshape(1, 3, new_h, new_w);
    std::vector<cv::Mat> input_channels;
    float* input_data = input_layer->mutable_cpu_data();
    for (int j = 0; j < input_layer->channels(); ++j) {
      cv::Mat channel(new_h, new_w, CV_32FC1, input_data);
      input_channels.push_back(channel);
      input_data += new_w * new_h;
    }
    cv::split(retImage, input_channels);
    pnet_->Forward();

    /* Copy the output layer to a std::vector */
    Blob* rect = pnet_->blob_by_name("conv4-2").get();
    Blob* prob = pnet_->blob_by_name("prob1").get();
    int feature_map_h = prob->shape(2);
    int feature_map_w = prob->shape(3);
    int count = prob->count() / 2;
    std::vector<float> bbox_regression;
    std::vector<float> probs;

    const float* rect_begin = rect->cpu_data();
    const float* rect_end = rect_begin + rect->channels() * count;
    bbox_regression = std::vector<float>(rect_begin, rect_end);

    const float* confidence_begin = prob->cpu_data() + count;
    const float* confidence_end = prob->cpu_data() + count * 2;

    probs = std::vector<float>(confidence_begin, confidence_end);
    std::vector<float> total_probs =
        std::vector<float>(prob->cpu_data(), prob->cpu_data() + prob->count());

    int stride = 2;
    int cellSize = cur_size;
    double scale = scales[i];
    int image_h = new_h;
    int image_w = new_w;
    for (int hh = 0; hh < feature_map_h; hh++) {
      for (int ww = 0; ww < feature_map_w; ww++) {
        int k = hh * feature_map_w + ww;
        if (probs[k] > pnet_thresh_) {
          confidences.push_back(probs[k]);
          cv::Rect bbox;
          float x1 = ((stride * ww + 1) / scale);
          float y1 = ((stride * hh + 1) / scale);
          float x2 = ((stride * ww + 1 + cellSize) / scale);
          float y2 = ((stride * hh + 1 + cellSize) / scale);
          int bbw = x2 - x1;
          int bbh = y2 - y1;
          int tx1 = x1;
          int ty1 = y1;
          x1 = tx1 + bbox_regression[k] * bbh;
          y1 = ty1 + bbox_regression[feature_map_h * feature_map_w +
                                     hh * feature_map_h + ww] *
                         bbw;
          x2 = tx1 + bbox_regression[2 * feature_map_h * feature_map_w +
                                     hh * feature_map_h + ww] *
                         bbh;
          y2 = ty1 + bbox_regression[3 * feature_map_h * feature_map_w +
                                     hh * feature_map_h + ww] *
                         bbw;
          bboxes.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        }
      }
    }

    NMS(bboxes, confidences, 0.4);
    cv::Mat img_show;
    image.copyTo(img_show);
    for (int k = 0; k < bboxes.size(); k++) {
      cv::rectangle(img_show, bboxes[k], cv::Scalar(0, 0, 255), 3);
      cv::putText(img_show, std::to_string(confidences[k]),
                  cvPoint(bboxes[k].x + 3, bboxes[k].y + 13),
                  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1,
                  CV_AA);
    }
    cv::imwrite("show_scale_" + std::to_string(scales[i]) + ".jpg", img_show);
  }
}

void MTCNN::RNetRun(const cv::Mat& image, std::vector<cv::Rect>& p_bboxes,
                    std::vector<float>& confidences,
                    std::vector<cv::Rect>& bboxes) {
  int idx = 0;
  for (auto p_box : p_bboxes) {
    cv::Mat img = crop(image, p_box);
    if (img.size() == cv::Size(0, 0)) continue;
    if (img.rows == 0 || img.cols == 0) continue;
    if (img.size() != rnet_input_size_) {
      cv::resize(img, img, rnet_input_size_);
    }
    cv::imwrite("rnet_" + std::to_string(idx++) + ".jpg", img);
    img.convertTo(img, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
    int new_h = rnet_input_size_.height;
    int new_w = rnet_input_size_.width;
    caffe::Blob* input_layer = rnet_->blob_by_name("data").get();
    input_layer->Reshape(1, 3, new_h, new_w);
    std::vector<cv::Mat> input_channels;
    float* input_data = input_layer->mutable_cpu_data();
    for (int j = 0; j < input_layer->channels(); ++j) {
      cv::Mat channel(new_h, new_w, CV_32FC1, input_data);
      input_channels.push_back(channel);
      input_data += new_w * new_h;
    }
    cv::split(img, input_channels);
    rnet_->Forward();

    /* Copy the output layer to a std::vector */
    Blob* rect = rnet_->blob_by_name("conv5-2").get();
    Blob* prob = rnet_->blob_by_name("prob1").get();
    const float* a = prob->cpu_data();
    if (prob->cpu_data()[1] > 0.35) {
      confidences.push_back(prob->cpu_data()[1]);
      int bbw = p_box.width;
      int bbh = p_box.height;
      float tx1 = p_box.x;
      float ty1 = p_box.y;
      int x1 = tx1 + rect->cpu_data()[0] * bbh;
      int y1 = ty1 + rect->cpu_data()[1] * bbw;
      int x2 = tx1 + rect->cpu_data()[2] * bbh;
      int y2 = ty1 + rect->cpu_data()[3] * bbw;
      bboxes.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    }
  }

  NMS(bboxes, confidences, 0.3);
  cv::Mat img_show;
  image.copyTo(img_show);
  for (int k = 0; k < bboxes.size(); k++) {
    cv::rectangle(img_show, bboxes[k], cv::Scalar(0, 0, 255), 3);
    cv::putText(img_show, std::to_string(confidences[k]),
                cvPoint(bboxes[k].x + 3, bboxes[k].y + 13),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1,
                CV_AA);
  }
  cv::imwrite("rnet_nms.jpg", img_show);
}

void MTCNN::ONetRun(const cv::Mat& image, std::vector<cv::Rect>& p_bboxes,
                    std::vector<float>& confidences,
                    std::vector<cv::Rect>& bboxes) {
  int idx = 0;
  for (auto p_box : p_bboxes) {
    cv::Mat img = crop(image, p_box);
    if (img.size() == cv::Size(0, 0)) continue;
    if (img.rows == 0 || img.cols == 0) continue;
    if (img.size() != onet_input_size_) {
      cv::resize(img, img, onet_input_size_);
    }
    cv::imwrite("onet_" + std::to_string(idx++) + ".jpg", img);
    img.convertTo(img, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
    int new_h = onet_input_size_.height;
    int new_w = onet_input_size_.width;
    caffe::Blob* input_layer = onet_->blob_by_name("data").get();
    input_layer->Reshape(1, 3, new_h, new_w);
    std::vector<cv::Mat> input_channels;
    float* input_data = input_layer->mutable_cpu_data();
    for (int j = 0; j < input_layer->channels(); ++j) {
      cv::Mat channel(new_h, new_w, CV_32FC1, input_data);
      input_channels.push_back(channel);
      input_data += new_w * new_h;
    }
    cv::split(img, input_channels);
    onet_->Forward();

    /* Copy the output layer to a std::vector */
    Blob* rect = onet_->blob_by_name("conv6-2").get();
    Blob* prob = onet_->blob_by_name("prob1").get();
    const float* a = prob->cpu_data();
    if (prob->cpu_data()[1] > 0.35) {
      confidences.push_back(prob->cpu_data()[1]);
      int bbw = p_box.width;
      int bbh = p_box.height;
      float tx1 = p_box.x;
      float ty1 = p_box.y;
      int x1 = tx1 + rect->cpu_data()[0] * bbh;
      int y1 = ty1 + rect->cpu_data()[1] * bbw;
      int x2 = tx1 + rect->cpu_data()[2] * bbh;
      int y2 = ty1 + rect->cpu_data()[3] * bbw;
      bboxes.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    }
  }

  NMS(bboxes, confidences, 0.3);
  cv::Mat img_show;
  image.copyTo(img_show);
  for (int k = 0; k < bboxes.size(); k++) {
    cv::rectangle(img_show, bboxes[k], cv::Scalar(0, 0, 255), 3);
    cv::putText(img_show, std::to_string(confidences[k]),
                cvPoint(bboxes[k].x + 3, bboxes[k].y + 13),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1,
                CV_AA);
  }
  cv::imwrite("onet_nms.jpg", img_show);
}

void MTCNN::NMS(std::vector<cv::Rect>& cur_rects,
                std::vector<float>& confidence, float threshold) {
  for (int i = 0; i < cur_rects.size(); i++) {
    for (int j = i + 1; j < cur_rects.size();) {
      if (IoU(cur_rects[i], cur_rects[j]) > threshold) {
        float a = IoU(cur_rects[i], cur_rects[j]);
        if (confidence[i] >= confidence[j] && confidence[j] < 0.96) {
          cur_rects.erase(cur_rects.begin() + j);
          confidence.erase(confidence.begin() + j);
        } else if (confidence[i] < confidence[j] && confidence[i] < 0.96) {
          cur_rects.erase(cur_rects.begin() + i);
          confidence.erase(confidence.begin() + i);
          i--;
          break;
        } else {
          j++;
        }
      } else {
        j++;
      }
    }
  }
}

void MTCNN::global_NMS() {
  std::vector<cv::Rect> cur_rects;
  std::vector<float> confidence;
  float threshold_IoM = threshold_NMS_;
  float threshold_IoU = threshold_NMS_ - 0.1;

  for (int i = 0; i < cur_rects.size(); i++) {
    for (int j = i + 1; j < cur_rects.size();) {
      if (IoU(cur_rects[i], cur_rects[j]) > threshold_IoU ||
          IoM(cur_rects[i], cur_rects[j]) > threshold_IoM) {
        if (confidence[i] >= confidence[j]) {
          cur_rects.erase(cur_rects.begin() + j);
          confidence.erase(confidence.begin() + j);
        } else if (confidence[i] < confidence[j]) {
          cur_rects.erase(cur_rects.begin() + i);
          confidence.erase(confidence.begin() + i);
          i--;
          break;
        } else {
          j++;
        }
      } else {
        j++;
      }
    }
  }
}

float MTCNN::IoU(cv::Rect rect1, cv::Rect rect2) {
  int x_overlap, y_overlap, intersection, unions;
  x_overlap =
      std::max(0, std::min((rect1.x + rect1.width), (rect2.x + rect2.width)) -
                      std::max(rect1.x, rect2.x));
  y_overlap =
      std::max(0, std::min((rect1.y + rect1.height), (rect2.y + rect2.height)) -
                      std::max(rect1.y, rect2.y));
  intersection = x_overlap * y_overlap;
  unions =
      rect1.width * rect1.height + rect2.width * rect2.height - intersection;
  return float(intersection) / unions;
}

float MTCNN::IoM(cv::Rect rect1, cv::Rect rect2) {
  int x_overlap, y_overlap, intersection, min_area;
  x_overlap =
      std::max(0, std::min((rect1.x + rect1.width), (rect2.x + rect2.width)) -
                      std::max(rect1.x, rect2.x));
  y_overlap =
      std::max(0, std::min((rect1.y + rect1.height), (rect2.y + rect2.height)) -
                      std::max(rect1.y, rect2.y));
  intersection = x_overlap * y_overlap;
  min_area =
      std::min((rect1.width * rect1.height), (rect2.width * rect2.height));
  return float(intersection) / min_area;
}

cv::Mat MTCNN::crop(cv::Mat img, cv::Rect& rect) {
  cv::Rect rect_old = rect;
  int stride = std::max(rect.width, rect.height);
  int center_x = rect.x + rect.width / 2;
  int center_y = rect.y + rect.height / 2;
  rect.x = center_x - stride / 2;
  rect.y = center_y - stride / 2;
  rect.width = stride;
  rect.height = stride;

  cv::Rect padding;

  if (rect.x < 0) {
    padding.x = -rect.x;
    rect.x = 0;
  }
  if (rect.y < 0) {
    padding.y = -rect.y;
    rect.y = 0;
  }
  if (img.cols < (rect.x + rect.width)) {
    padding.width = rect.x + rect.width - img.cols;
    rect.width = img.cols - rect.x;
  }
  if (img.rows < (rect.y + rect.height)) {
    padding.height = rect.y + rect.height - img.rows;
    rect.height = img.rows - rect.y;
  }
  if (rect.width < 0 || rect.height < 0) {
    rect = cv::Rect(0, 0, 0, 0);
    padding = cv::Rect(0, 0, 0, 0);
  }
  cv::Mat img_cropped = img(rect);
  if (padding.x || padding.y || padding.width || padding.height) {
    cv::copyMakeBorder(img_cropped, img_cropped, padding.y, padding.height,
                       padding.x, padding.width, cv::BORDER_CONSTANT,
                       cv::Scalar(0));
    // here, the rect should be changed
    rect.x -= padding.x;
    rect.y -= padding.y;
    rect.width += padding.width + padding.x;
    rect.width += padding.height + padding.y;
  }

  return img_cropped;
}
