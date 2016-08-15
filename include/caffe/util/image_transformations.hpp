#ifndef CAFFE_UTIL_IMAGE_TRANSFORMATIONS_H_
#define CAFFE_UTIL_IMAGE_TRANSFORMATIONS_H_

namespace caffe {

#ifdef USE_OPENCV
cv::Mat ResizeImagePeriodic(const cv::Mat& src_img,
    const int off_h, const int offset_w, , cv::Mat& dst_img);

#endif  // USE_OPENCV

}  // namespace caffe

#endif  // CAFFE_UTIL_IMAGE_TRANSFORMATIONS_H_