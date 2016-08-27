#ifndef CAFFE_SOFTMAX_WITH_ROC_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAX_WITH_ROC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

/**
 * @brief Computes 1-area under the receiver operating cahracteristic curve
 * This layer only accept channel number 2
 * At test time, this layer can be replaced simply by a SoftmaxLayer.
 *
 * @param bottom input Blob vector (length 2)
 * @param top output Blob vector (length 1)
 */
template <typename Dtype>
class SoftmaxWithROCLossLayer : public LossLayer<Dtype> {
 public:
  explicit SoftmaxWithROCLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxWithROCLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void SetIsPositiveNegative(const int count, const Dtype* label);
  virtual void ProbToPosNeg();
  virtual void PosNegToProb();

  /// The internal SoftmaxLayer used to map predictions to a distribution.
  shared_ptr<Layer<Dtype> > softmax_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;
  /// softmax dimension
  int softmax_axis_, outer_num_, inner_num_;
  /// negative id and positive id
  Dtype positive_label_, negative_label_;
  /// linear region approximation for backprop
  Dtype eps_;
  /// blobs that whether a prediction is postive or not
  Blob<Dtype> is_positive_;
  Blob<Dtype> is_negative_;
  /// hold reorgaized probs for both positive and negative
  Blob<Dtype> prob_positive_;
  Blob<Dtype> prob_negative_;
  /// blobs that stores the diff between postive and negative
  Blob<Dtype> diff_;
  /// ones
  Blob<Dtype> ones_;
  /// normalizer
  int normalizer_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_ROC_LOSS_LAYER_HPP_
