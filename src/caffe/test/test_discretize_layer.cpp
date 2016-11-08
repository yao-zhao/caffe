#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/discretize_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class DiscretizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  DiscretizeLayerTest()
      : blob_bottom_(new Blob<Dtype>(8, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    Dtype* bottom_data = blob_bottom_->mutable_cpu_data();
    for (int i = 0; i < 8; ++i) {
      bottom_data[i] = Dtype(i);
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DiscretizeLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DiscretizeLayerTest, TestDtypesAndDevices);


TYPED_TEST(DiscretizeLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_discretize_param()->add_separator(4);
  DiscretizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(DiscretizeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_discretize_param()->add_separator(-2.);
  layer_param.mutable_discretize_param()->add_separator(2.5);
  layer_param.mutable_discretize_param()->add_separator(5.4);
  DiscretizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_->cpu_data();
  EXPECT_EQ(top_data[0], 1.);
  EXPECT_EQ(top_data[1], 1.);
  EXPECT_EQ(top_data[2], 1.);
  EXPECT_EQ(top_data[3], 2.);
  EXPECT_EQ(top_data[4], 2.);
  EXPECT_EQ(top_data[5], 2.);
  EXPECT_EQ(top_data[6], 3.);
  EXPECT_EQ(top_data[7], 3.);
}

}  // namespace caffe
