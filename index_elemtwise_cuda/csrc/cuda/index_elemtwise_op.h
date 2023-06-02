#include <torch/extension.h>
#include <ATen/Dispatch.h>

const std::map<c10::ScalarType, int> scalarTypeSize {
    {torch::ScalarType::Byte, sizeof(uint8_t)},
    {torch::ScalarType::Char, sizeof(int8_t)},
    {torch::ScalarType::Short, sizeof(int16_t)},
    {torch::ScalarType::Int, sizeof(int)},
    {torch::ScalarType::Long, sizeof(int64_t)},
    {torch::ScalarType::Float, sizeof(float)},
    {torch::ScalarType::Half, sizeof(c10::Half)},
    {torch::ScalarType::BFloat16, sizeof(c10::BFloat16)},
    {torch::ScalarType::Double, sizeof(double)},
    {torch::ScalarType::Bool, sizeof(bool)},
    {torch::ScalarType::QInt8, sizeof(c10::qint8)},
    {torch::ScalarType::QUInt8, sizeof(c10::quint8)},
    {torch::ScalarType::QInt32,sizeof(c10::qint32)},
    {torch::ScalarType::QUInt4x2, sizeof(c10::quint4x2)},
    {torch::ScalarType::QUInt2x4, sizeof(c10::quint2x4)},
};

torch::Tensor index_elemwise_op_launch(torch::Tensor src1, torch::Tensor src2,
                                       torch::Tensor src_idx1, torch::Tensor src_idx2,
                                       std::string operate_name);

std::tuple<torch::Tensor, torch::Tensor>
index_elemwise_op_backward_launch(torch::Tensor grad_out, torch::Tensor src1, torch::Tensor src2,
                                  torch::Tensor src_idx1, torch::Tensor src_idx2,
                                  std::string operate_name);