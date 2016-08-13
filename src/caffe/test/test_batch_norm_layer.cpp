#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 2
#define INPUT_DATA_SIZE 3

namespace caffe {

  template <typename TypeParam>
  class BatchNormLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
  protected:

    BatchNormLayerTest()
        : blob_bottom_(new Blob<Dtype>(4, 3, 2, 5)),
          blob_top_(new Blob<Dtype>()),
          blob_bottom2_(new Blob<Dtype>()),
          blob_param_(new Blob<Dtype>()) {
      // fill the values
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
      int num = this->blob_bottom_->num();
      int channels = this->blob_bottom_->channels();
      int height = this->blob_bottom_->height();
      int width = this->blob_bottom_->width();
      blob_bottom2_->ReshapeLike(*blob_bottom_);
      // fill the values
      for (int j = 0; j < channels; ++j) {
        int index =0;
        for (int i = 0; i < num; ++i) {
          for ( int k = 0; k < height; ++k ) {
            for ( int l = 0; l < width; ++l ) {
              index++;
              this->blob_bottom2_->mutable_cpu_data()
              [this->blob_bottom2_->offset(i, j, k, l)]
              = Dtype(2*(j+3) + 2*(index%2)-1);
            }
          }
        }
      }
    }
    virtual ~BatchNormLayerTest() { delete blob_bottom_; delete blob_top_;
      delete blob_bottom2_; delete blob_param_; }
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    Blob<Dtype>* const blob_bottom2_;
    Blob<Dtype>* const blob_param_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
  };

  TYPED_TEST_CASE(BatchNormLayerTest, TestDtypesAndDevices);

  TYPED_TEST(BatchNormLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    BatchNormLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Test mean
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();

    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0;
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = this->blob_top_->data_at(i, j, k, l);
            sum += data;
            var += data * data;
          }
        }
      }
      sum /= height * width * num;
      var /= height * width * num;

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
    }
  }

  TYPED_TEST(BatchNormLayerTest, TestTwoTopsForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    BatchNormParameter* batch_norm_param =
      layer_param.mutable_batch_norm_param();
    batch_norm_param->set_use_global_stats(false);
    this->blob_top_vec_.push_back(this->blob_param_);

    BatchNormLayer<Dtype> layer(layer_param);
    this->blob_bottom_vec_[0] = this->blob_bottom2_;
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->blob_bottom_vec_[0] = this->blob_bottom_;

    // Test mean
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();

    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0;
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = this->blob_top_->data_at(i, j, k, l);
            sum += data;
            var += data * data;
          }
        }
      }
      sum /= height * width * num;
      var /= height * width * num;

      const Dtype kErrorBound = 0.1;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
      // expect  mean and var
      const Dtype* batch_norm_data = this->blob_top_vec_[1]->cpu_data();
      EXPECT_NEAR(2*(j+3), batch_norm_data[j], kErrorBound);
      EXPECT_NEAR(1, batch_norm_data[j+channels], kErrorBound);
    }
  }

  TYPED_TEST(BatchNormLayerTest, TestForwardInplace) {
    typedef typename TypeParam::Dtype Dtype;
    Blob<Dtype> blob_inplace;
    blob_inplace.ReshapeLike(*this->blob_bottom_);
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    LayerParameter layer_param;
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_inplace);
    blob_bottom_vec.push_back(&blob_inplace);
    blob_top_vec.push_back(&blob_inplace);

    BatchNormLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec, blob_top_vec);
    layer.Forward(blob_bottom_vec, blob_top_vec);

    // Test mean
    int num = blob_inplace.num();
    int channels = blob_inplace.channels();
    int height = blob_inplace.height();
    int width = blob_inplace.width();

    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0;
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = blob_inplace.data_at(i, j, k, l);
            sum += data;
            var += data * data;
          }
        }
      }
      sum /= height * width * num;
      var /= height * width * num;

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
    }
  }

  TYPED_TEST(BatchNormLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

    BatchNormLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-4);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

}  // namespace caffe
