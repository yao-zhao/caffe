#ifndef CAFFE_LORENTZIAN_PROB_LOSS_LAYER_HPP_
#define CAFFE_LORENTZIAN_PROB_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief assume the prediction of a label follows
 * a lorentzian distribution
 * use predicted mean and predicted sigma
 * to maximize the probability of observing
 * the labels
 * first bottom blob accept predicted label mean
 * second bottom blob accept predicted label sigma
 * third bottom blob accept the label
 */

template <typename Dtype>
class LorentzianProbLossLayer : public LossLayer<Dtype> {
 public:
  explicit LorentzianProbLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "GaussianProbLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * do not allow to backprop to bottom[2] which is label
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    if (bottom_index < 2) {
      return true;
    }
    return false;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> tmp_;
  Dtype eps_;
  Blob<Dtype> sumvec_;
};

}  // namespace caffe
#endif  // CAFFE_LORENTZIAN_PROB_LOSS_LAYER_HPP_
