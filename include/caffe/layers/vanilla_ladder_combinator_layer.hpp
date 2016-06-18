#ifndef CAFFE_VANILLA_LADDER_COMBINATOR_LAYER_HPP_
#define CAFFE_VANILLA_LADDER_COMBINATOR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief vanilla ladder combinator function g(\tilde z^l,u^{l+1}) = 
 * b_0 + w_{0z} \odot \tilde z^l + w_{0u}\odot u^{l+1} +w_{0zu}\odot\tilde z^l \odot u^{l+1} + 
 * w_{\sigma}\odot Sigmoid(b_1+w_{1z}\odot\tilde z^l+w_{1u}\odot u^{l+1}+w_{1zu}\odot\tilde z^l\odot u^{l+1})
 * 
 * the combinator functions combines hidden variables from noisy pass \tilde z^l and reconstruction pass u^{l+1}
 */
template <typename Dtype>
class VanillaLadderCombinatorLayer : public Layer<Dtype> {
 public:
  explicit VanillaLadderCombinatorLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VanillaLadderCombinator"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // EltwiseParameter_EltwiseOp op_;
  // vector<Dtype> coeffs_;
  // Blob<int> max_idx_;
  int axis_;
  int outer_dim_, inner_dim_, comb_dim_;
  Blob<Dtype> temp_;
  Blob<Dtype> tempmul_;
  Blob<Dtype> tempsig_;

  // bool stable_prod_grad_;
};

}  // namespace caffe

#endif  // CAFFE_ELTWISE_LAYER_HPP_
