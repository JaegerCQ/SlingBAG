#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

constexpr float K_COEF = 1.0E6F;
constexpr float RSQRT_OF_PI = 0.5641895835477563F;

// 定义参数数组
__constant__ float dx_factors[10] = {0.5f, 0.6f, 0.9f, 1.2f, 1.5f, 1.8f, 2.1f, 2.4f, 2.7f, 3.0f};
__constant__ float p0_factors[10] = {10.0f/55.0f, 9.0f/55.0f, 8.0f/55.0f, 7.0f/55.0f, 
                                    6.0f/55.0f, 5.0f/55.0f, 4.0f/55.0f, 3.0f/55.0f, 
                                    2.0f/55.0f, 1.0f/55.0f};

// ======================== 设备函数 ========================
__device__ __forceinline__ float smooth_step(const float x) {
    return 0.5F * (1.0F + erff(K_COEF * x));
}

__device__ __forceinline__ float smooth_delta(const float x) {
    return K_COEF * RSQRT_OF_PI * expf(-K_COEF * K_COEF * x * x);
}

__device__ __forceinline__ float signal_set_radius(
    const float dx, const float p0, const float t, const float xr, const float yr, const float zr, 
    const float xs, const float ys, const float zs) {
    const float Rx = xr - xs;
    const float Ry = yr - ys;
    const float Rz = zr - zs;
    const float R = sqrtf(fmaf(Rx, Rx, fmaf(Ry, Ry, Rz * Rz)));
    const float D = fmaf(-1500.0F, t, R);
    return p0 * D / (2.0F * R) * (smooth_step(D + dx) - smooth_step(D - dx));
}

// ======================== 前向传播核函数 ========================
__global__ void simulate_cuda_kernel(
    const float *__restrict__ sensor_location,
    const float *__restrict__ source_location,
    const float *__restrict__ source_p0,
    const float *__restrict__ source_dx,
    const float dt,
    const size_t num_sensors,
    const size_t num_sources,
    const size_t num_times,
    float *__restrict__ simulate_record) {

    // 共享内存分配（每个 block 处理 blockDim.x 个 source）
    extern __shared__ float shared_data[];
    float *shared_loc = shared_data;
    float *shared_p0 = shared_data + blockDim.x * 3;
    float *shared_dx = shared_p0 + blockDim.x;

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sensors * num_times) return;

    const size_t sensor_idx = idx / num_times;
    const float t = (idx % num_times) * dt;
    const float xr = sensor_location[sensor_idx * 3 + 0];
    const float yr = sensor_location[sensor_idx * 3 + 1];
    const float zr = sensor_location[sensor_idx * 3 + 2];

    float result = 0.0f;

    // 分块处理所有 source
    for (size_t src_block = 0; src_block < num_sources; src_block += blockDim.x) {
        const size_t src_idx = src_block + threadIdx.x;
        if (src_idx < num_sources) {
            // 协作加载当前块的 source 数据到共享内存
            shared_loc[threadIdx.x * 3 + 0] = source_location[src_idx * 3 + 0];
            shared_loc[threadIdx.x * 3 + 1] = source_location[src_idx * 3 + 1];
            shared_loc[threadIdx.x * 3 + 2] = source_location[src_idx * 3 + 2];
            shared_p0[threadIdx.x] = source_p0[src_idx];
            shared_dx[threadIdx.x] = source_dx[src_idx];
        }
        __syncthreads();

        // 计算当前块的所有 source 贡献
        const size_t block_size = (blockDim.x < (num_sources - src_block)) ? blockDim.x : (num_sources - src_block);
        for (size_t i = 0; i < block_size; i++) {
            const float xs = shared_loc[i * 3 + 0];
            const float ys = shared_loc[i * 3 + 1];
            const float zs = shared_loc[i * 3 + 2];
            const float p0 = shared_p0[i];
            const float dx = shared_dx[i];

            // 计算 10 个 factor 的贡献
            result += signal_set_radius(dx_factors[0] * dx, p0_factors[0] * p0, t, xr, yr, zr, xs, ys, zs);
            result += signal_set_radius(dx_factors[1] * dx, p0_factors[1] * p0, t, xr, yr, zr, xs, ys, zs);
            result += signal_set_radius(dx_factors[2] * dx, p0_factors[2] * p0, t, xr, yr, zr, xs, ys, zs);
            result += signal_set_radius(dx_factors[3] * dx, p0_factors[3] * p0, t, xr, yr, zr, xs, ys, zs);
            result += signal_set_radius(dx_factors[4] * dx, p0_factors[4] * p0, t, xr, yr, zr, xs, ys, zs);
            result += signal_set_radius(dx_factors[5] * dx, p0_factors[5] * p0, t, xr, yr, zr, xs, ys, zs);
            result += signal_set_radius(dx_factors[6] * dx, p0_factors[6] * p0, t, xr, yr, zr, xs, ys, zs);
            result += signal_set_radius(dx_factors[7] * dx, p0_factors[7] * p0, t, xr, yr, zr, xs, ys, zs);
            result += signal_set_radius(dx_factors[8] * dx, p0_factors[8] * p0, t, xr, yr, zr, xs, ys, zs);
            result += signal_set_radius(dx_factors[9] * dx, p0_factors[9] * p0, t, xr, yr, zr, xs, ys, zs);
        }
        __syncthreads();
    }

    simulate_record[idx] = result;
}

// ======================== 前向传播接口 ========================
torch::Tensor simulate_cuda(
    const torch::Tensor sensor_location,
    const torch::Tensor source_location,
    const torch::Tensor source_p0,
    const torch::Tensor source_dx,
    const float dt,
    const size_t num_sensors,
    const size_t num_sources,
    const size_t num_times) {

    auto simulate_record = torch::zeros({static_cast<int64_t>(num_sensors * num_times)}, sensor_location.options());

    // 计算合适的 block 大小（例如 256 线程/block）
    const size_t thread_per_block = 256;
    const size_t num_blocks = (num_sensors * num_times + thread_per_block - 1) / thread_per_block;

    // 计算共享内存大小（每个 block 需要存储 blockDim.x 个 source 的数据）
    const size_t shared_mem_size = thread_per_block * (3 + 1 + 1) * sizeof(float);

    simulate_cuda_kernel<<<num_blocks, thread_per_block, shared_mem_size>>>(
        sensor_location.data_ptr<float>(),
        source_location.data_ptr<float>(),
        source_p0.data_ptr<float>(),
        source_dx.data_ptr<float>(),
        dt,
        num_sensors,
        num_sources,
        num_times,
        simulate_record.data_ptr<float>());

    cudaDeviceSynchronize();
    return simulate_record;
}

// ======================== 反向传播核函数 ========================
__global__ void simulate_cuda_backward_kernel(
    const float *__restrict__ sensor_location,
    const float *__restrict__ source_location,
    const float *__restrict__ source_p0,
    const float *__restrict__ source_dx,
    const float *__restrict__ dL_dsimulate_record,
    const float dt,
    const size_t num_sensors,
    const size_t num_sources,
    const size_t num_times,
    float *__restrict__ grad_source_location,
    float *__restrict__ grad_source_p0,
    float *__restrict__ grad_source_dx) {

    const size_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (source_idx >= num_sources) return;

    const float xs = source_location[source_idx * 3 + 0];
    const float ys = source_location[source_idx * 3 + 1];
    const float zs = source_location[source_idx * 3 + 2];
    const float dx = source_dx[source_idx];
    const float p0 = source_p0[source_idx];

    float grad_xs = 0.0f, grad_ys = 0.0f, grad_zs = 0.0f;
    float grad_p0 = 0.0f, grad_dx = 0.0f;

    // 遍历所有 sensor 和 time
    for (size_t idx = 0; idx < num_sensors * num_times; idx++) {
        const size_t sensor_idx = idx / num_times;
        const float t = (idx % num_times) * dt;
        const float xr = sensor_location[sensor_idx * 3 + 0];
        const float yr = sensor_location[sensor_idx * 3 + 1];
        const float zr = sensor_location[sensor_idx * 3 + 2];
        const float dL = dL_dsimulate_record[idx];

        // 计算梯度贡献（10 个 factor 的梯度累加）
        for (int i = 0; i < 10; i++) {
            const float A = dx_factors[i] * dx;
            const float P = p0_factors[i] * p0;
            const float D = fmaf(-1500.0F, t, sqrtf(fmaf(xr-xs, xr-xs, fmaf(yr-ys, yr-ys, (zr-zs)*(zr-zs)))));
            const float R = sqrtf(fmaf(xr-xs, xr-xs, fmaf(yr-ys, yr-ys, (zr-zs)*(zr-zs))));

            // 计算 partial_signal_set_radius_partial_* 的梯度贡献
            grad_xs += dL * (-(xr-xs)/R) * (P * D / (2.0F * R) * (smooth_delta(D + A) - smooth_delta(D - A)) +
                                         750.0F * P * t / (R * R) * (smooth_step(D + A) - smooth_step(D - A)));
            grad_ys += dL * (-(yr-ys)/R) * (P * D / (2.0F * R) * (smooth_delta(D + A) - smooth_delta(D - A)) +
                                         750.0F * P * t / (R * R) * (smooth_step(D + A) - smooth_step(D - A)));
            grad_zs += dL * (-(zr-zs)/R) * (P * D / (2.0F * R) * (smooth_delta(D + A) - smooth_delta(D - A)) +
                                         750.0F * P * t / (R * R) * (smooth_step(D + A) - smooth_step(D - A)));
            grad_p0 += dL * p0_factors[i] * (D / (2.0F * R) * (smooth_step(D + A) - smooth_step(D - A)));
            grad_dx += dL * dx_factors[i] * (P * D / (2.0F * R) * (smooth_delta(D + A) + smooth_delta(D - A)));
        }
    }

    // 原子更新梯度（避免 race condition）
    // atomicAdd((double*)(&grad_source_location[source_idx * 3 + 0]), (double)grad_xs);
    atomicAdd(&grad_source_location[source_idx * 3 + 0], grad_xs);
    atomicAdd(&grad_source_location[source_idx * 3 + 1], grad_ys);
    atomicAdd(&grad_source_location[source_idx * 3 + 2], grad_zs);
    atomicAdd(&grad_source_p0[source_idx], grad_p0);
    atomicAdd(&grad_source_dx[source_idx], grad_dx);
}

// ======================== 反向传播接口 ========================
std::vector<torch::Tensor> simulate_cuda_backward(
    const torch::Tensor sensor_location,
    const torch::Tensor source_location,
    const torch::Tensor source_p0,
    const torch::Tensor source_dx,
    const torch::Tensor dL_dsimulate_record,
    const float dt,
    const size_t num_sensors,
    const size_t num_sources,
    const size_t num_times) {

    auto grad_source_location = torch::zeros({static_cast<int64_t>(num_sources * 3)}, source_location.options());
    auto grad_source_p0 = torch::zeros({static_cast<int64_t>(num_sources)}, source_p0.options());
    auto grad_source_dx = torch::zeros({static_cast<int64_t>(num_sources)}, source_dx.options());

    const size_t thread_per_block = 256;
    const size_t num_blocks = (num_sources + thread_per_block - 1) / thread_per_block;

    simulate_cuda_backward_kernel<<<num_blocks, thread_per_block>>>(
        sensor_location.data_ptr<float>(),
        source_location.data_ptr<float>(),
        source_p0.data_ptr<float>(),
        source_dx.data_ptr<float>(),
        dL_dsimulate_record.data_ptr<float>(),
        dt,
        num_sensors,
        num_sources,
        num_times,
        grad_source_location.data_ptr<float>(),
        grad_source_p0.data_ptr<float>(),
        grad_source_dx.data_ptr<float>());

    cudaDeviceSynchronize();
    return {grad_source_location, grad_source_p0, grad_source_dx};
}
