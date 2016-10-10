#ifndef CAFFE_REGRESSION_LABEL_LAYER_HPP_
#define CAFFE_REGRESSION_LABEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief shift the label space into two inverse spaces
 *    this is usefull in regression task to balance the feature weights
 *    bottom blob accept labels
 *    top blob 0 shift the labels by top[0] = bottom[0] - lower_bound()
 *    top blob 1 shift the labels by top[1] = upper_bround() - bottom[0]
 */
template <typename Dtype>
class RegressionLabelLayer : public Layer<Dtype> {
 public:
  explicit RegressionLabelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RegressionLabel"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype lb_;
  Dtype ub_;
};

}  // namespace caffe

#endif  // CAFFE_REGRESSION_LABEL_LAYER_HPP_
