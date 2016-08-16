#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

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

}  // namespace caffe

#endif  // USE_OPENCV
