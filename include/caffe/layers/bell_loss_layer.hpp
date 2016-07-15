#ifndef CAFFE_BELL_LOSS_LAYER_HPP_
#define CAFFE_BELL_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief euclidean distance and then 
 * normalize it with bell curve
 */
template <typename Dtype>
class BellLossLayer : public LossLayer<Dtype> {
 public:
  explicit BellLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BellLoss"; }
  /**
   * do not allow to backprop to bottom[1] which is label
   */
  // virtual inline bool AllowForceBackward(const int bottom_index) const {
  //   return true;
  // }

 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Dtype power_;
  Dtype threshold_;
};

}  // namespace caffe

#endif  // CAFFE_BELL_LOSS_LAYER_HPP_
