#ifndef CAFFE_LADDER_LOSS_LAYER_HPP_
#define CAFFE_LADDER_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the euclidean distance between clean path z and reconstructed (\hat{z}-\mu)/\sigma
 * Euclidean (L2) loss @f$
 *         E = \frac{1}{2N*C*H*W} \sum\limits_{n=1}^N \left| \left| (\hat{z}_n -\mu)/\sigma- z_n
 *        \right| \right|_2^2 @f$ 
 *
 * @param bottom input Blob vector (length 3)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the reconstructed variable @f$ z \in [-\infty, +\infty]@f$
*   -# @f$ (N \times C \times H \times W) @f$
 *      the clean path variable z @f$ \hat{z} \in [-\infty, +\infty]@f$
 *   -# @f$ (2 * C)
        the mean and variance of each channel

 * @param top output Blob vector (length 1)
 * Euclidean (L2) loss @f$
 *         E = \frac{1}{2N*C*H*W} \sum\limits_{n=1}^N \left| \left| (\hat{z}_n -\mu)/\sigma- z_n
 *        \right| \right|_2^2 @f$ 
 *
 */
template <typename Dtype>
class LadderLossLayer : public LossLayer<Dtype> {
 public:
  explicit LadderLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LadderLoss"; }
  /**
   * Unlike most loss layers, in the EuclideanLossLayer we can backpropagate
   * to both inputs of z and \hat{z} but not the mean and variance
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 2;
  }

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactTopBlobs() const { return 1; }


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
  Blob<Dtype> batch_sum_multiplier_;
  Blob<Dtype> num_by_chans_;
  Blob<Dtype> spatial_sum_multiplier_;
  Blob<Dtype> tempvar_;
  int channels_;

};

}  // namespace caffe

#endif  // CAFFE_LADDER_LOSS_LAYER_HPP_
