#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/common.hpp"
#include "caffe/util/image_transformations.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

void ResizeImagePeriodic(const cv::Mat& src_img,
    const int h_off, const int w_off, cv::Mat* dst_img) {
  const int h_src = src_img.rows;
  const int w_src = src_img.cols;
  const int h_dst = dst_img->rows;
  const int w_dst = dst_img->cols;
  const int num_channels = src_img.channels();
  CHECK_EQ(num_channels, dst_img->channels()) <<
    "number of channels of source and destimation images have to be equal";
  CHECK_NE(dst_img, &src_img) << "doesn't support in place calculation";
  const int cvwidth_src = w_src * num_channels;
  for (int h = 0; h < h_dst; ++h) {
    uchar* ptr_dst = dst_img->ptr<uchar>(h);
    const uchar* ptr_src = src_img.ptr<uchar>(positive_mod(h-h_off, h_src));
    int index_dst = 0;
    int index_src = positive_mod(-num_channels*w_off, cvwidth_src);
    for (int w = 0; w < w_dst; ++w) {
      for (int c = 0; c < num_channels; ++c) {
        ptr_dst[index_dst++] = ptr_src[positive_mod(index_src++, cvwidth_src)];
      }
    }
  }
}

void ResizeImagePeriodicMirror(const cv::Mat& src_img,
    const int h_off, const int w_off, cv::Mat* dst_img) {
  const int h_src = src_img.rows;
  const int w_src = src_img.cols;
  const int h_dst = dst_img->rows;
  const int w_dst = dst_img->cols;
  const int num_channels = src_img.channels();
  CHECK_EQ(num_channels, dst_img->channels()) <<
    "number of channels of source and destimation images have to be equal";
  CHECK_NE(dst_img, &src_img) << "doesn't support in place calculation";
  for (int h = 0; h < h_dst; ++h) {
    uchar* ptr_dst = dst_img->ptr<uchar>(h);
    const uchar* ptr_src = src_img.ptr<uchar>(
        (floor_div(h-h_off, h_src))%2 == 0 ? positive_mod(h-h_off, h_src) :
        h_src-1-positive_mod(h-h_off, h_src));
    int index_dst = 0;
    for (int w = 0; w < w_dst; ++w) {
      int index_src = positive_mod(w-w_off, w_src);
      index_src = (floor_div(w-w_off, w_src))%2 == 0 ?
          index_src : w_src-1-index_src;
      index_src *= num_channels;
      for (int c = 0; c < num_channels; ++c) {
        ptr_dst[index_dst++] = ptr_src[index_src++];
      }
    }
  }
}

void RandomRotateImage(const cv::Mat& src_img, const int rotation_range,
    const float rescale_factor, cv::Mat* dst_img) {
  CHECK_GT(rescale_factor, 0);
  CHECK_GE(rotation_range, 0);
  CHECK_LE(rotation_range, 360);
  cv::Point2f pt(src_img.cols/2, src_img.rows/2);
  cv::Mat r;
  if (rotation_range != 0) {
    r = cv::getRotationMatrix2D(pt,
        caffe_rng_rand()%rotation_range-rotation_range/2, rescale_factor);
  } else {
    r = cv::getRotationMatrix2D(pt, 0, rescale_factor);
  }
  cv::warpAffine(src_img, *dst_img, r, cv::Size(src_img.cols, src_img.rows));
}

void RandomPerspectiveTransformImage(const cv::Mat& src_img,
    const int perspective_transformation_border, cv::Mat* dst_img) {
  CHECK_GE(perspective_transformation_border, 0);
  CHECK_LE(perspective_transformation_border, src_img.rows/2);
  CHECK_LE(perspective_transformation_border, src_img.cols/2);
  cv::Point2f src_shape[4];
  src_shape[0] = cv::Point2f(0+
      caffe_rng_rand()%perspective_transformation_border,
      0+caffe_rng_rand()%perspective_transformation_border);
  src_shape[1] = cv::Point2f(0+
      caffe_rng_rand()%perspective_transformation_border,
      src_img.rows-caffe_rng_rand()%perspective_transformation_border);
  src_shape[2] = cv::Point2f(src_img.cols
      -caffe_rng_rand()%perspective_transformation_border,
      src_img.rows-caffe_rng_rand()%perspective_transformation_border);
  src_shape[3] = cv::Point2f(src_img.cols
      -caffe_rng_rand()%perspective_transformation_border,
      0+caffe_rng_rand()%perspective_transformation_border);
  cv::Point2f dst_shape[4];
  dst_shape[0] = cv::Point2f(0, 0);
  dst_shape[1] = cv::Point2f(0, src_img.rows);
  dst_shape[2] = cv::Point2f(src_img.cols, src_img.rows);
  dst_shape[3] = cv::Point2f(src_img.cols, 0);
  cv::Mat ptmatrix = cv::getPerspectiveTransform(src_shape, dst_shape);
  cv::warpPerspective(src_img, *dst_img, ptmatrix,
      cv::Size(src_img.cols, src_img.cols),
      cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}

}  // namespace caffe

#endif  // USE_OPENCV
