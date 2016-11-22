#ifndef CAFFE_YOLO_LOSS_LAYER_HPP_
#define CAFFE_YOLO_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

/**
 * @brief Computes the YOLO loss
 *
 *
 * bottom[0] accept bounding box predictions, N * S_h * S_w * B * (x,y,w,h,c)
 * bottom[1] accept labels of bounding boxes, N * maxnumbox * (x,y,w,h,C)
 * bottom[2] optionally accept class labels for, N * S_h * S_w * C,
 *    need to be softmaxed
 */
template <typename Dtype>
class YoloLossLayer : public LossLayer<Dtype> {
 public:
  explicit YoloLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "YoloLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
// virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
// virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// yolo params
  int S_h_, S_w_, B_, C_;
  /// label parameter
  int maxnumbox_;
  /// wether thas third bottom, has class label data
  bool has_class_prob_;
  /// weight modifier
  Dtype lambda_coord_, lambda_noobj_;
  /// grid label
  Blob<Dtype> grid_label_;
  /// minimal denominator for sqrt w and h
  Dtype eps_;
  /// diff
  Blob<Dtype> diff_;
};

}  // namespace caffe

#endif  // CAFFE_YOLO_LOSS_LAYER_HPP_
