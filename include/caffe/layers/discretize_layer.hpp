#ifndef CAFFE_DISCRETIZE_LAYER_HPP_
#define CAFFE_DISCRETIZE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Tests whether the input exceeds a threshold: outputs label for inputs
 *        between label regions
 */
template <typename Dtype>
class DiscretizeLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ThresholdParameter threshold_param,
   *     with ThresholdLayer options:
   *   - separator (\b repeated).
   */
  explicit DiscretizeLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Discretize"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /// @brief Not implemented (non-differentiable function)
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
  int num_separators_;
  Blob<Dtype> separators_;
};

}  // namespace caffe

#endif  // CAFFE_DISCRETIZE_LAYER_HPP_
