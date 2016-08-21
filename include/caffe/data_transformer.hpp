#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum> & datum_vector,
                Blob<Dtype>* transformed_blob);

#ifdef USE_OPENCV
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<cv::Mat> & mat_vector,
                Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);
    /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   * works for data and its pixel label
   * mirror and crop operations apply the same time
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob_img
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob_img,
      const cv::Mat& cv_lab, Blob<Dtype>* transformed_blob_lab);
#endif  // USE_OPENCV

  /**
   * @brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   */
#ifdef USE_OPENCV
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const cv::Mat& cv_img);
#endif  // USE_OPENCV

 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);
   /**
   * @brief Check if no advanced transformations parameters are set
   * advanced transformations all require USE_OPENCV
   */
  void NoAdvancedTransformations();
   /**
   * @brief get the correct height and width after cropping and also compare
   *    them with height width of input
   *
   * @param crop_h
   *    cropped height
   * @param crop_w
   *    cropped width
   * @param height
   *    input height
   * @param width
   *    input width
   * @return
   *    if cropping is valid, has_crop
   */
  bool GetPostCropSize(int* crop_h, int* crop_w,
      const int height, const int width);
   /**
   * @brief get the mean of the image, saved in data_mean, mean_values_
   *
   * @param height
   *    input height
   * @param width
   *    input width
   * @param channels
   *    input channels
   */
  void GetMean(const int channels,
      const int height, const int width);
   /**
   * @brief get the offset of crop
   *
   * @param off_h
   *    offset height
   * @param off_w
   *    offset width
   * @param crop_h
   *    cropped height
   * @param crop_w
   *    cropped width
   * @param height
   *    input height
   * @param width
   *    input width
   */
  void GetOffset(int* off_h, int* off_w,
    const int crop_h, const int crop_w, const int height, const int width);
   /**
   * @brief get the factor to scale the intensity of the image
   *
   * @return
   *    get the image intensity scale factor, contrast jitter included
   */
  Dtype GetScale();
#ifdef USE_OPENCV
   /**
   * @brief apply geometrical transformations include rotation, scale jitter
   *    and perspetcitive transformations
   *
   * @param cv_img
   *    image to be transformed
   */
  void GeometricalTransform(cv::Mat* cv_img);
   /**
   * @brief resize an image using periodic boundary condition
   *
   * @param input_height
   *    input height after resize
   * @param width
   *    input width after resize
   * @param cv_img
   *    image to be resize
   */
  void PeriodicResize(cv::Mat* cv_img, int* input_height, int* input_width);
#endif  // USE_OPENCV

  void Transform(const Datum& datum, Dtype* transformed_data);
  // Tranformation parameters
  TransformationParameter param_;

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
