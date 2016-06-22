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
 *         E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| (\hat{z}_n -\mu)/\sigma- z_n
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
 *         E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| (\hat{z}_n -\mu)/\sigma- z_n
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

  /**
   * @brief Computes the Euclidean error gradient w.r.t. the inputs.
   *
   * Unlike other children of LossLayer, EuclideanLossLayer \b can compute
   * gradients with respect to the label inputs bottom[1] (but still only will
   * if propagate_down[1] is set, due to being produced by learnable parameters
   * or if force_backward is set). In fact, this layer is "commutative" -- the
   * result is the same regardless of the order of the two bottoms.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$\hat{y}@f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial \hat{y}} =
   *            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
   *      @f$ if propagate_down[0]
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets @f$y@f$; Backward fills their diff with gradients
   *      @f$ \frac{\partial E}{\partial y} =
   *          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
   *      @f$ if propagate_down[1]
   */
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
