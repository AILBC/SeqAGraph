#include "index_elemtwise_op.h"
#include "kernel_fun.cuh"

#define IdxElemLaunch(name, ...)\
    switch (OperateDefine.at(name)) {\
        case Sum: {\
            idxSum(__VA_ARGS__);\
            break;\
        }\
        case Mul: {\
            idxMul(__VA_ARGS__);\
            break;\
        }\
    }\

#define IdxElemBwdLaunch(name, ...)\
    switch (OperateDefine.at(name)) {\
        case Sum: {\
            idxSumBwd(__VA_ARGS__);\
            break;\
        }\
        case Mul: {\
            idxMulBwd(__VA_ARGS__);\
            break;\
        }\
    }\

torch::Tensor index_elemwise_op_launch(torch::Tensor src1, torch::Tensor src2,
                                       torch::Tensor src_idx1, torch::Tensor src_idx2,
                                       std::string operate_name)
{
    INPUT_CHECKING(src1);
    INPUT_CHECKING(src2);
    INPUT_CHECKING(src_idx1);
    if (src_idx2.numel() > 0)
        INPUT_CHECKING(src_idx2);

    auto out_size = src1.sizes().vec();
    out_size[0] = src_idx1.size(0);

    torch::Tensor out = torch::empty(out_size, src1.options());

    auto M = out.size(0);
    int64_t D = 1;
    for (auto i = 1; i < out.dim(); ++i)
        D *= out.size(i);
    
    bool vectorize = false;
    if (
        ((out.scalar_type() == torch::ScalarType::Float) && (D % 4 == 0)) |
        ((out.scalar_type() == torch::ScalarType::Half) && (D % 2 == 0)) |
        ((out.scalar_type() == torch::ScalarType::BFloat16) && (D % 2 == 0))
    )
        vectorize = true;
    
    AT_DISPATCH_ALL_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::BFloat16, out.scalar_type(), "index_elemtwise_operation_forward", [&] {
        auto src_data1 = src1.data_ptr<scalar_t>();
        auto src_data2 = src2.data_ptr<scalar_t>();
        auto out_data = out.data_ptr<scalar_t>();
        auto idx_info1 = src_idx1.data_ptr<int64_t>();
        auto out_numel = out.numel();
        
        int64_t* idx_info2 = src_idx2.numel() > 0 ? src_idx2.data_ptr<int64_t>() : nullptr;

        IdxElemLaunch(
            operate_name,
            src_data1, src_data2, idx_info1,
            idx_info2, out_data, D, out_numel, vectorize
        );
    });
    return out;
}

std::tuple<torch::Tensor, torch::Tensor>
index_elemwise_op_backward_launch(torch::Tensor grad_out, torch::Tensor src1, torch::Tensor src2,
                                  torch::Tensor src_idx1, torch::Tensor src_idx2,
                                  std::string operate_name)
{
    INPUT_CHECKING(src1);
    INPUT_CHECKING(src2);
    INPUT_CHECKING(grad_out);
    INPUT_CHECKING(src_idx1);
    if (src_idx2.numel() > 0)
        INPUT_CHECKING(src_idx2);

    torch::Tensor grad_src1 = torch::zeros(src1.sizes().vec(), grad_out.options());
    torch::Tensor grad_src2 = torch::zeros(src2.sizes().vec(), grad_out.options());

    auto M = grad_out.size(0);
    int64_t D = 1;
    for (auto i = 1; i < grad_out.dim(); ++i)
        D *= grad_out.size(i);
    
    bool vectorize = false;
    if (
        ((grad_out.scalar_type() == torch::ScalarType::Half) && (D % 2 == 0)) |
        ((grad_out.scalar_type() == torch::ScalarType::BFloat16) && (D % 2 == 0))
    )
        vectorize = true;

    AT_DISPATCH_ALL_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::BFloat16, grad_out.scalar_type(), "index_elemtwise_operation_backward", [&] {
        auto grad_out_data = grad_out.data_ptr<scalar_t>();
        auto src_data1 = src1.data_ptr<scalar_t>();
        auto src_data2 = src2.data_ptr<scalar_t>();
        auto grad_src_data1 = grad_src1.data_ptr<scalar_t>();
        auto grad_src_data2 = grad_src2.data_ptr<scalar_t>();
        auto idx_info1 = src_idx1.data_ptr<int64_t>();
        auto out_numel = grad_out.numel();

        int64_t* idx_info2 = src_idx2.numel() > 0 ? src_idx2.data_ptr<int64_t>() : nullptr;

        IdxElemBwdLaunch(
            operate_name,
            grad_out_data, idx_info1, idx_info2,
            src_data1, src_data2, grad_src_data1,
            grad_src_data2, D, out_numel, vectorize
        );
    });
    return std::make_tuple(grad_src1, grad_src2);
}