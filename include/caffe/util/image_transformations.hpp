#ifndef CAFFE_UTIL_IMAGE_TRANSFORMATIONS_H_
#define CAFFE_UTIL_IMAGE_TRANSFORMATIONS_H_
#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>

#include <vector>

#include "caffe/common.hpp"

namespace caffe {

  /**
   * @brief Resize the image periodically
   *
   * @param src_img, dst_img
   *    cv::Mat contains source image and testination image
   * @param off_h, off_w
   *    the (0,0) pixel coordinates of src_img in dst_img
   */
void ResizeImagePeriodic(const cv::Mat& src_img,
    const int off_h, const int off_w, cv::Mat* dst_img);

  /**
   * @brief Resize the image periodically but with mirror condition
   *    so that the pixels are still continous in resized image
   *
   * @param src_img, dst_img
   *    cv::Mat contains source image and testination image
   * @param off_h, off_w
   *    the (0,0) pixel coordinates of src_img in dst_img
   */
void ResizeImagePeriodicMirror(const cv::Mat& src_img,
    const int off_h, const int off_w, cv::Mat* dst_img);

  /**
   * @brief Randomly Rotate the image from the center
   *    and rescale the image bu tje factor rescale_factor
   *
   * @param src_img, dst_img
   *    cv::Mat contains source image and testination image
   * @param rotation_range
   *    rotation range of the image [0, 360]
   * @param rescale_factor
   *    factor to rescale the image by, usually set at 1 for no scales
   */
void RandomRotateImage(const cv::Mat& src_img, const int rotation_range,
    const float rescale_factor, const vector<int> & border_value,
    cv::Mat* dst_img);

  /**
   * @brief use random perspective transformation on the image
   *
   * @param src_img, dst_img
   *    cv::Mat contains source image and testination image
   * @param perspective_transformation_border
   *    the max random displace from the four corners of the images
   */
void RandomPerspectiveTransformImage(const cv::Mat& src_img,
    const int perspective_transformation_border, cv::Mat* dst_img);

}  // namespace caffe

#endif  // USE_OPENCV
#endif  // CAFFE_UTIL_IMAGE_TRANSFORMATIONS_H_
