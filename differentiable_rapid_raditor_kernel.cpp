#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)

// Forward declaration of CUDA functions
torch::Tensor simulate_cuda(
    const torch::Tensor sensor_location, const torch::Tensor source_location,
    const torch::Tensor source_p0, const torch::Tensor source_dx, const float dt, const size_t num_sensors,
    const size_t num_sources, const size_t num_times);

std::vector<torch::Tensor> simulate_cuda_backward(
    const torch::Tensor sensor_location, const torch::Tensor source_location, const torch::Tensor source_p0, 
    const torch::Tensor source_dx, const torch::Tensor dL_dsimulate_record, const float dt, const size_t num_sensors, 
    const size_t num_sources, const size_t num_times);



// Entry point for PyTorch
torch::Tensor simulate(
    const torch::Tensor sensor_location, const torch::Tensor source_location,
    const torch::Tensor source_p0, const torch::Tensor source_dx, const float dt, const size_t num_sensors,
    const size_t num_sources, const size_t num_times)
{
    CHECK_INPUT(sensor_location);
    CHECK_INPUT(source_location);
    CHECK_INPUT(source_p0);
    CHECK_INPUT(source_dx);
    return simulate_cuda(
        sensor_location, source_location, source_p0, source_dx, dt, num_sensors, num_sources, num_times);
}

std::vector<torch::Tensor> simulate_backward(
    const torch::Tensor sensor_location, const torch::Tensor source_location, const torch::Tensor source_p0, 
    const torch::Tensor source_dx, const torch::Tensor dL_dsimulate_record, const float dt, const size_t num_sensors, 
    const size_t num_sources, const size_t num_times)
{
    CHECK_INPUT(sensor_location);
    CHECK_INPUT(source_location);
    CHECK_INPUT(source_p0);
    CHECK_INPUT(source_dx);
    CHECK_INPUT(dL_dsimulate_record);
    
    return simulate_cuda_backward(
        sensor_location, source_location, source_p0, source_dx, dL_dsimulate_record, dt, num_sensors, num_sources, num_times
        );
    
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("simulate", &simulate, "Simulate wave propagation");
    m.def("simulate_backward", &simulate_backward, "Backward gradient of simulate function");
}