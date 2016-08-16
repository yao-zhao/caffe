#ifndef CAFFE_UTIL_IMAGE_TRANSFORMATIONS_H_
#define CAFFE_UTIL_IMAGE_TRANSFORMATIONS_H_

#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>

namespace caffe {

void ResizeImagePeriodic(const cv::Mat& src_img,
    const int off_h, const int off_w, cv::Mat& dst_img);

#endif  // USE_OPENCV

}  // namespace caffe

#endif  // CAFFE_UTIL_IMAGE_TRANSFORMATIONS_H_