#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/yolo_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class YoloLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  YoloLossLayerTest()
      : blob_bottom_box_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_bottom_class_(new Blob<Dtype>()),
        blob_top_loss_(new Blob<Dtype>()) {
    // for (int i = 0; i < blob_bottom_label_->count(); ++i) {
    //   blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    // }
    // for (int i = 0; i < blob_bottom_label_->count(); ++i) {
    //   blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    // }
    // blob_bottom_vec_.push_back(blob_bottom_box_);
    // blob_bottom_vec_.push_back(blob_bottom_label_);
    // blob_top_vec_.push_back(blob_top_loss_);
  }
  void ManualSetup1() {
    int box_dim[] = {1, 3, 3, 1, 5};
    blob_bottom_box_->Reshape(*(new vector<int>(box_dim, box_dim+5)));
    Dtype input1[] = {.5, .5, .5, .5, 1.0,
                      .5, .5, .5, .5, 1.0,
                      .5, .5, .5, .5, 1.0,
                      .5, .5, .5, .5, 1.0,
                      .5, .5, .5, .5, 1.0,
                      .5, .5, .5, .5, 1.0,
                      .5, .5, .5, .5, 1.0,
                      .5, .5, .5, .5, 1.0,
                      .5, .5, .5, .5, 1.0,
                     };
    for (int i = 0; i < blob_bottom_box_->count(); ++i) {
      blob_bottom_box_->mutable_cpu_data()[i] = input1[i];
    }
    int label_dim[] = {1, 2, 5};
    blob_bottom_label_->Reshape(*(new vector<int>(label_dim, label_dim+3)));
    Dtype input2[] = {2/3., 2/3., 2/3., 2/3., 0,
                      .5, .5, .5, .5, -1,
                     };
    blob_bottom_vec_.clear();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = input2[i];
    }
    blob_bottom_vec_.push_back(blob_bottom_box_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.clear();
    blob_top_vec_.push_back(blob_top_loss_);
  }
  void ManualSetup2() {
    int box_dim[] = {1, 3, 3, 1, 5};
    blob_bottom_box_->Reshape(*(new vector<int>(box_dim, box_dim+5)));
    Dtype input1[] = {.5, .5, .5, .5, .8,
                      .5, .5, .5, .5, .8,
                      .5, .5, .5, .5, .8,
                      .5, .5, .5, .5, .8,
                      .5, .5, .5, .5, .8,
                      .5, .5, .5, .5, .8,
                      .5, .5, .5, .5, .8,
                      .5, .5, .5, .5, .8,
                      .5, .5, .5, .5, .8,
                     };
    for (int i = 0; i < blob_bottom_box_->count(); ++i) {
      blob_bottom_box_->mutable_cpu_data()[i] = input1[i];
    }
    int label_dim[] = {1, 2, 5};
    blob_bottom_label_->Reshape(*(new vector<int>(label_dim, label_dim+3)));
    Dtype input2[] = {.51, .51, .25, .25, 0,
                      .5, .5, .1, .1, -1,
                     };
    blob_bottom_vec_.clear();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = input2[i];
    }
    blob_bottom_vec_.push_back(blob_bottom_box_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.clear();
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~YoloLossLayerTest() {
    delete blob_bottom_box_;
    delete blob_bottom_label_;
    delete blob_bottom_class_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_box_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_class_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(YoloLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(YoloLossLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  YoloLossLayer<Dtype> layer(layer_param);
  this->ManualSetup1();
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(YoloLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_yolo_loss_param()->set_lambda_coord(5.0);
  layer_param.mutable_yolo_loss_param()->set_lambda_noobj(0.25);
  YoloLossLayer<Dtype> layer(layer_param);
  this->ManualSetup2();
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype loss = 8*0.25*pow(0.8, 2)/2.+5.0*(2*pow(sqrt(.5)-sqrt(.25), 2)
    +pow(0.8-0.25, 2)+2*pow(0.01, 2))/2.;
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
}

TYPED_TEST(YoloLossLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_yolo_loss_param()->set_lambda_coord(5.0);
  layer_param.mutable_yolo_loss_param()->set_lambda_noobj(0.25);
  YoloLossLayer<Dtype> layer(layer_param);
  this->ManualSetup2();
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  this->blob_top_loss_->mutable_cpu_diff()[0] = 2;
  vector<bool> propagate_down;
  propagate_down.push_back(true);
  propagate_down.push_back(false);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  // for (int i = 0; i < this->blob_bottom_box_->count(0, 4); ++i) {
  //   for (int j = 0; j < 5; ++j) {
  //     cout << " " << this->blob_bottom_box_->cpu_diff()[5*i+j] << " ";
  //   }
  //   cout << "\n";
  // }
}

// gradient test shouldn't work: because ground truth confidence
// is calculated by iou, so that label is input dependent.
// but we don't want the net to create a label that is more wrong
// so that the result can be equally wrong as predicted
// so estimated gradient is wrong! confidence error backproped to x,y,w,h
// TYPED_TEST(YoloLossLayerTest, TestGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   layer_param.mutable_yolo_loss_param()->set_lambda_coord(5.0);
//   layer_param.mutable_yolo_loss_param()->set_lambda_noobj(0.25);
//   YoloLossLayer<Dtype> layer(layer_param);
//   this->ManualSetup2();
//   GradientChecker<Dtype> checker(1e-3, 1e-3, 1701);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//       this->blob_top_vec_, 0);
// }

}  // namespace caffe
