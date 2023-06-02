#include "cuda/index_elemtwise_op.h"

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

torch::Tensor idx_check(torch::optional<Variable> src_idx, int64_t src_num, torch::Device src_device) {
    auto idx = src_idx.has_value() ? src_idx.value() : torch::arange(src_num).to(src_device);

    // if (idx.max().cpu().data_ptr<int64_t>()[0] > src_num)
    //     AT_ERROR("Maximum index must be small or equal with input Tensors size");
    
    return idx;
}

void tgt_size_check(torch::Tensor src_idx1, torch::Tensor src_idx2) {
    if (src_idx1.numel() != src_idx2.numel())
        AT_ERROR("Two index must in the same size");
}

void src_check(variable_list src) {
    if (src.size() != 2)
        AT_ERROR("Number of input Tensors must equal to 2");
}

c10::ScalarType out_type_check(torch::Tensor src1, torch::Tensor src2) {
    c10::ScalarType out_type;
    if (src1.scalar_type() == src2.scalar_type()) {
        out_type = src1.scalar_type();
    }
    else {
        out_type = scalarTypeSize.at(src1.scalar_type()) < scalarTypeSize.at(src2.scalar_type()) ? src2.scalar_type() : src1.scalar_type();
    }
    return out_type;
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<int64_t>, std::vector<int64_t>>
broadcast(torch::Tensor src1, torch::Tensor src2) {
    if (src1.dim() != src2.dim())
        AT_ERROR("the diminsion of inputs tenosor must be the same");

    auto broadcast_size1 = src1.sizes().vec();
    auto broadcast_size2 = src2.sizes().vec();
    std::vector<int64_t> broadcast_dim1, broadcast_dim2;

    for (auto i = src1.dim() - 1; i >= 1; --i) {
        auto dim_size = src1.size(i) < src2.size(i) ? src2.size(i) : src1.size(i);
        if (broadcast_size1[i] != dim_size) {
            broadcast_size1[i] = dim_size;
            broadcast_dim1.push_back(i);
        }
        if (broadcast_size2[i] != dim_size) {
            broadcast_size2[i] = dim_size;
            broadcast_dim2.push_back(i);
        }
        
    }
    auto broadcast_src1 = src1.expand(broadcast_size1).contiguous();
    auto broadcast_src2 = src2.expand(broadcast_size2).contiguous();
    return std::make_tuple(broadcast_src1, broadcast_src2, broadcast_dim1, broadcast_dim2);
}

// euqal to operation in torch: src1.index_select(0, idx1) ~ src2.index_select(0, idx2)
// src[src1, src2], src_idx[idx1, idx2(optional)]
// src[i]->(N, d_model1, d_model2, ...), idx[i].dim() = 1 (index for N)
class IdxElemwiseOP: public torch::autograd::Function<IdxElemwiseOP> {
    public:
    static variable_list forward(
        AutogradContext *ctx,
        Variable src1, Variable src2,
        Variable src_idx1, torch::optional<Variable> src_idx2,
        std::string operate_name
    ) {
        Variable src_idx2_val = torch::empty({0});
        if (src_idx2.has_value()) {
            src_idx2_val = src_idx2.value();
            tgt_size_check(src_idx1, src_idx2_val);
        }

        c10::ScalarType out_type = out_type_check(src1, src2);
        src1 = src1.to(out_type);
        src2 = src2.to(out_type);

        ctx->saved_data["operate_name"] = operate_name;
        ctx->save_for_backward({src1, src2, src_idx1, src_idx2_val});

        auto out = index_elemwise_op_launch(src1, src2, src_idx1, src_idx2_val, operate_name);
        return {out};
    }

    static variable_list backward(
        AutogradContext *ctx, variable_list grad_outs
    ) {
        auto grad_out = grad_outs[0].contiguous();
        auto saved = ctx->get_saved_variables();
        auto src1 = saved[0];
        auto src2 = saved[1];
        auto idx1 = saved[2];
        auto idx2 = saved[3];
        auto operate_name = ctx->saved_data["operate_name"].toStringRef();

        src1 = src1.to(grad_out.scalar_type());
        src2 = src2.to(grad_out.scalar_type());
        auto grad_inputs = index_elemwise_op_backward_launch(grad_out, src1, src2, idx1, idx2, operate_name);
        auto grad_in1 = std::get<0>(grad_inputs);
        auto grad_in2 = std::get<1>(grad_inputs);
        return {grad_in1, grad_in2, Variable(), Variable(), Variable()};
    }
};

torch::Tensor index_elemwise_operate(
    torch::Tensor src1, torch::Tensor src2,
    torch::Tensor src_idx1, torch::optional<torch::Tensor> src_idx2,
    std::string operate_name
) {
    return IdxElemwiseOP::apply(src1, src2, src_idx1, src_idx2, operate_name)[0];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("IndexElemtwiseOperate", &index_elemwise_operate, "Index Elementwise Operation (CUDA)");
}