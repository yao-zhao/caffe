#ifndef CAFFE_SOFTMAX_WITH_DECAY_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAX_WITH_DECAY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
class SoftmaxWithDecayLossLayer : public SoftmaxWithLossLayer<Dtype> {
 public:
  explicit SoftmaxWithDecayLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxWithDecayLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int method_;
  int softmax_dim_;
  Dtype rate_;
  Blob<Dtype> weights_;
  Blob<Dtype> label_idx_;
  Blob<Dtype> mid_inner_multiplier_;

};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_DECAY_LOSS_LAYER_HPP_
