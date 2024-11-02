#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

constexpr float K_COEF = 1.0E6F;
constexpr float RSQRT_OF_PI = 0.5641895835477563F;
constexpr float BOUNDARY = 1.0E-5F;

__device__ __forceinline__ float smooth_step(const float x)
{
    if (x > BOUNDARY)
    {
        return 1.0F;
    }
    else if (x < -BOUNDARY)
    {
        return 0.0F;
    }
    else
    {
        return 0.5F * (1.0F + erff(K_COEF * x));
    }
}

__device__ __forceinline__ float smooth_delta(const float x)
{
    if (fabsf(x) > BOUNDARY)
    {
        return 0.0F;
    }
    else
    {
        return K_COEF * RSQRT_OF_PI * expf(-K_COEF * K_COEF * x * x);
    }
}

__device__ __forceinline__ float signal_set_radius(
    const float dx, const float p0, const float t, const float xr, const float yr, const float zr, const float xs,
    const float ys, const float zs)
{
    const float Rx = xr - xs;
    const float Ry = yr - ys;
    const float Rz = zr - zs;
    const float R = sqrtf(fmaf(Rx, Rx, fmaf(Ry, Ry, Rz * Rz)));
    const float D = fmaf(-1500.0F, t, R);
    return p0 * D / (2.0F * R) * (smooth_step(D + dx) - smooth_step(D - dx));
}

__global__ void simulate_cuda_kernel(
    const float *sensor_location, const float *source_location, const float *source_p0, const float *source_dx,
    const float dt, const size_t num_sensors, const size_t num_sources, const size_t num_times, float *simulate_record)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t sensor_idx = idx / num_times;
    const float t = (idx % num_times) * dt;
    if (idx < num_sensors * num_times)
    {
        const float xr = sensor_location[sensor_idx * 3 + 0];
        const float yr = sensor_location[sensor_idx * 3 + 1];
        const float zr = sensor_location[sensor_idx * 3 + 2];
        for (size_t source_idx = 0; source_idx < num_sources; ++source_idx)
        {
            const float xs = source_location[source_idx * 3 + 0];
            const float ys = source_location[source_idx * 3 + 1];
            const float zs = source_location[source_idx * 3 + 2];
            const float p0 = source_p0[source_idx];
            const float dx = source_dx[source_idx];
            simulate_record[idx] +=
                (signal_set_radius(0.5F * dx, 10.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 signal_set_radius(0.6F * dx, 9.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 signal_set_radius(0.9F * dx, 8.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 signal_set_radius(1.2F * dx, 7.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 signal_set_radius(1.5F * dx, 6.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 signal_set_radius(1.8F * dx, 5.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 signal_set_radius(2.1F * dx, 4.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 signal_set_radius(2.4F * dx, 3.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 signal_set_radius(2.7F * dx, 2.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 signal_set_radius(3.0F * dx, 1.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs));
        }
    }
}

// Function to call forward CUDA kernel
torch::Tensor simulate_cuda(
    const torch::Tensor sensor_location, const torch::Tensor source_location, const torch::Tensor source_p0,
    const torch::Tensor source_dx, const float dt, const size_t num_sensors, const size_t num_sources,
    const size_t num_times)
{
    auto simulate_record = torch::zeros({static_cast<int64_t>(num_sensors * num_times)}, sensor_location.options());

    const size_t thread_per_block = 256;
    const size_t num_blocks = (num_sensors * num_times + thread_per_block - 1) / thread_per_block;

    simulate_cuda_kernel<<<num_blocks, thread_per_block>>>(
        sensor_location.data_ptr<float>(), source_location.data_ptr<float>(), source_p0.data_ptr<float>(),
        source_dx.data_ptr<float>(), dt, num_sensors, num_sources, num_times, simulate_record.data_ptr<float>());
    cudaDeviceSynchronize();

    // catch error
    // cudaError_t error = cudaGetLastError();
    // printf("CUDA error: %s\n", cudaGetErrorString(error));

    return simulate_record;
}

// sensor_num: 400
// source_num: 1000
// sensor_location: torch.Size([400, 3])
// source_location: torch.Size([1000, 3])
// source_p0: torch.Size([1000])
// radius_0: torch.Size([1000])
// simulate_record: torch.Size([1638400]) 4096*400

// source_num: 1000
// grad_source_location: torch.Size([1000, 3])
// grad_source_p0: torch.Size([1000])
// grad_radius_0: torch.Size([1000])

__device__ __forceinline__ float partial_signal_set_radius_partial_P(
    const float A, const float P, const float t, const float xr, const float yr, const float zr, const float xs,
    const float ys, const float zs)
{
    const float Rx = xr - xs;
    const float Ry = yr - ys;
    const float Rz = zr - zs;
    const float R = sqrtf(fmaf(Rx, Rx, fmaf(Ry, Ry, Rz * Rz)));
    const float D = fmaf(-1500.0F, t, R);

    // return (D / (2 * R)) * (1.0F / (1.0F + expf(-10000.0F * (D + A))) - 1.0F / (1.0F + expf(-10000.0F * (D - A))));

    // const float u = expf(-10000.0F * (D + A));
    // const float v = expf(-10000.0F * (D - A));
    // return D / (2.0F * R) * (1.0F / (1.0F + u) - 1.0F / (1.0F + v));

    return D / (2.0F * R) * (smooth_step(D + A) - smooth_step(D - A));
}

__device__ __forceinline__ float partial_signal_set_radius_partial_A(
    const float A, const float P, const float t, const float xr, const float yr, const float zr, const float xs,
    const float ys, const float zs)
{
    const float Rx = xr - xs;
    const float Ry = yr - ys;
    const float Rz = zr - zs;
    const float R = sqrtf(fmaf(Rx, Rx, fmaf(Ry, Ry, Rz * Rz)));
    const float D = fmaf(-1500.0F, t, R);

    // return (P * D / (2 * R)) *
    //        ((10000.0F * expf(-10000.0F * (D + A))) /
    //             ((1.0F + expf(-10000.0F * (D + A))) * (1.0F + expf(-10000.0F * (D + A)))) +
    //         (10000.0F  * expf(-10000.0F * (D - A))) /
    //             ((1.0F + expf(-10000.0F * (D - A))) * (1.0F + expf(-10000.0F * (D - A)))));

    // const float u = expf(-10000.0F * (D + A));
    // const float v = expf(-10000.0F * (D - A));
    // return 10000.0F * P * D / (2.0F * R) * (1.0F / (u + 1.0F / u + 2.0F) + 1.0F / (v + 1.0F / v + 2.0F));

    return P * D / (2.0F * R) * (smooth_delta(D + A) + smooth_delta(D - A));
}

__device__ __forceinline__ float partial_signal_set_radius_partial_xs(
    const float A, const float P, const float t, const float xr, const float yr, const float zr, const float xs,
    const float ys, const float zs)
{
    const float Rx = xr - xs;
    const float Ry = yr - ys;
    const float Rz = zr - zs;
    const float R = sqrtf(fmaf(Rx, Rx, fmaf(Ry, Ry, Rz * Rz)));
    const float D = fmaf(-1500.0F, t, R);
    // return (-Rx / R) *
    //        ((P * D / (2 * R)) * ((10000.0F * expf(-10000.0F * (D + A))) /
    //                                  ((1.0F + expf(-10000.0F * (D + A))) * (1.0F + expf(-10000.0F * (D + A)))) -
    //                              (10000.0F * expf(-10000.0F * (D - A))) /
    //                                  ((1.0F + expf(-10000.0F * (D - A))) * (1.0F + expf(-10000.0F * (D - A))))) +
    //         (P * 750.0F * t / (R * R)) *
    //             (1.0F / (1.0F + expf(-10000.0F * (D + A))) - 1.0F / (1.0F + expf(-10000.0F * (D - A)))));
    const float u = expf(-10000.0F * (D + A));
    const float v = expf(-10000.0F * (D - A));
    return -Rx / R *
           (10000.0F * P * D / (2.0F * R) * (1.0F / (u + 1.0F / u + 2.0F) - 1.0F / (v + 1.0F / v + 2.0F)) +
            750.0F * P * t / (R * R) * (1.0F / (1.0F + u) - 1.0F / (1.0F + v)));
}

__device__ __forceinline__ float partial_signal_set_radius_partial_ys(
    const float A, const float P, const float t, const float xr, const float yr, const float zr, const float xs,
    const float ys, const float zs)
{
    const float Rx = xr - xs;
    const float Ry = yr - ys;
    const float Rz = zr - zs;
    const float R = sqrtf(fmaf(Rx, Rx, fmaf(Ry, Ry, Rz * Rz)));
    const float D = fmaf(-1500.0F, t, R);
    // return (-Ry / R) *
    //        ((P * D / (2 * R)) * ((10000.0F * expf(-10000.0F * (D + A))) /
    //                                  ((1.0F + expf(-10000.0F * (D + A))) * (1.0F + expf(-10000.0F * (D + A)))) -
    //                              (10000.0F * expf(-10000.0F * (D - A))) /
    //                                  ((1.0F + expf(-10000.0F * (D - A))) * (1.0F + expf(-10000.0F * (D - A))))) +
    //         (P * 750.0F * t / (R * R)) *
    //             (1.0F / (1.0F + expf(-10000.0F * (D + A))) - 1.0F / (1.0F + expf(-10000.0F * (D - A)))));
    const float u = expf(-10000.0F * (D + A));
    const float v = expf(-10000.0F * (D - A));
    return -Ry / R *
           (10000.0F * P * D / (2.0F * R) * (1.0F / (u + 1.0F / u + 2.0F) - 1.0F / (v + 1.0F / v + 2.0F)) +
            750.0F * P * t / (R * R) * (1.0F / (1.0F + u) - 1.0F / (1.0F + v)));
}

__device__ __forceinline__ float partial_signal_set_radius_partial_zs(
    const float A, const float P, const float t, const float xr, const float yr, const float zr, const float xs,
    const float ys, const float zs)
{
    const float Rx = xr - xs;
    const float Ry = yr - ys;
    const float Rz = zr - zs;
    const float R = sqrtf(fmaf(Rx, Rx, fmaf(Ry, Ry, Rz * Rz)));
    const float D = fmaf(-1500.0F, t, R);
    // return (-Rz / R) *
    //        ((P * D / (2 * R)) * ((10000.0F * expf(-10000.0F * (D + A))) /
    //                                  ((1.0F + expf(-10000.0F * (D + A))) * (1.0F + expf(-10000.0F * (D + A)))) -
    //                              (10000.0F * expf(-10000.0F * (D - A))) /
    //                                  ((1.0F + expf(-10000.0F * (D - A))) * (1.0F + expf(-10000.0F * (D - A))))) +
    //         (P * 750.0F * t / (R * R)) *
    //             (1.0F / (1.0F + expf(-10000.0F * (D + A))) - 1.0F / (1.0F + expf(-10000.0F * (D - A)))));
    const float u = expf(-10000.0F * (D + A));
    const float v = expf(-10000.0F * (D - A));
    return -Rz / R *
           (10000.0F * P * D / (2.0F * R) * (1.0F / (u + 1.0F / u + 2.0F) - 1.0F / (v + 1.0F / v + 2.0F)) +
            750.0F * P * t / (R * R) * (1.0F / (1.0F + u) - 1.0F / (1.0F + v)));
}

__global__ void simulate_cuda_backward_kernel(
    const float *sensor_location, const float *source_location, const float *source_p0, const float *source_dx,
    const float *dL_dsimulate_record, const float dt, const size_t num_sensors, const size_t num_sources,
    const size_t num_times, float *grad_source_location, float *grad_source_p0, float *grad_source_dx)
{
    const size_t source_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 线程引索判断计算在内层
    if (source_idx < num_sources)
    {
        const float xs = source_location[source_idx * 3 + 0];
        const float ys = source_location[source_idx * 3 + 1];
        const float zs = source_location[source_idx * 3 + 2];
        const float dx = source_dx[source_idx];
        const float p0 = source_p0[source_idx];

        for (size_t idx = 0; idx < num_sensors * num_times; ++idx)
        {
            size_t sensor_idx = idx / num_times;
            float t = (idx % num_times) * dt;
            const float xr = sensor_location[sensor_idx * 3 + 0];
            const float yr = sensor_location[sensor_idx * 3 + 1];
            const float zr = sensor_location[sensor_idx * 3 + 2];
            float partial_simulate_record_idx_partial_p0 =
                (10.0F / 55.0F *
                     partial_signal_set_radius_partial_P(0.5F * dx, 10.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 9.0F / 55.0F *
                     partial_signal_set_radius_partial_P(0.6F * dx, 9.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 8.0F / 55.0F *
                     partial_signal_set_radius_partial_P(0.9F * dx, 8.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 7.0F / 55.0F *
                     partial_signal_set_radius_partial_P(1.2F * dx, 7.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 6.0F / 55.0F *
                     partial_signal_set_radius_partial_P(1.5F * dx, 6.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 5.0F / 55.0F *
                     partial_signal_set_radius_partial_P(1.8F * dx, 5.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 4.0F / 55.0F *
                     partial_signal_set_radius_partial_P(2.1F * dx, 4.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 3.0F / 55.0F *
                     partial_signal_set_radius_partial_P(2.4F * dx, 3.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 2.0F / 55.0F *
                     partial_signal_set_radius_partial_P(2.7F * dx, 2.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 1.0F / 55.0F *
                     partial_signal_set_radius_partial_P(3.0F * dx, 1.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs));

            float partial_simulate_record_idx_partial_a0 =
                (0.5F * partial_signal_set_radius_partial_A(0.5F * dx, 10.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 0.6F * partial_signal_set_radius_partial_A(0.6F * dx, 9.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 0.9F * partial_signal_set_radius_partial_A(0.9F * dx, 8.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 1.2F * partial_signal_set_radius_partial_A(1.2F * dx, 7.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 1.5F * partial_signal_set_radius_partial_A(1.5F * dx, 6.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 1.8F * partial_signal_set_radius_partial_A(1.8F * dx, 5.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 2.1F * partial_signal_set_radius_partial_A(2.1F * dx, 4.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 2.4F * partial_signal_set_radius_partial_A(2.4F * dx, 3.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 2.7F * partial_signal_set_radius_partial_A(2.7F * dx, 2.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
                 3.0F * partial_signal_set_radius_partial_A(3.0F * dx, 1.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs));

            // float partial_simulate_record_idx_partial_xs =
            //     (partial_signal_set_radius_partial_xs(0.5F * dx, 10.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_xs(0.6F * dx, 9.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_xs(0.9F * dx, 8.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_xs(1.2F * dx, 7.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_xs(1.5F * dx, 6.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_xs(1.8F * dx, 5.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_xs(2.1F * dx, 4.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_xs(2.4F * dx, 3.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_xs(2.7F * dx, 2.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_xs(3.0F * dx, 1.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs));

            // float partial_simulate_record_idx_partial_ys =
            //     (partial_signal_set_radius_partial_ys(0.5F * dx, 10.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_ys(0.6F * dx, 9.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_ys(0.9F * dx, 8.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_ys(1.2F * dx, 7.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_ys(1.5F * dx, 6.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_ys(1.8F * dx, 5.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_ys(2.1F * dx, 4.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_ys(2.4F * dx, 3.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_ys(2.7F * dx, 2.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_ys(3.0F * dx, 1.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs));

            // float partial_simulate_record_idx_partial_zs =
            //     (partial_signal_set_radius_partial_zs(0.5F * dx, 10.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_zs(0.6F * dx, 9.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_zs(0.9F * dx, 8.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_zs(1.2F * dx, 7.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_zs(1.5F * dx, 6.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_zs(1.8F * dx, 5.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_zs(2.1F * dx, 4.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_zs(2.4F * dx, 3.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_zs(2.7F * dx, 2.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs) +
            //      partial_signal_set_radius_partial_zs(3.0F * dx, 1.0F / 55.0F * p0, t, xr, yr, zr, xs, ys, zs));

            // grad_source_location[source_idx * 3 + 0] +=
            //     partial_simulate_record_idx_partial_xs * dL_dsimulate_record[idx];
            // grad_source_location[source_idx * 3 + 1] +=
            //     partial_simulate_record_idx_partial_ys * dL_dsimulate_record[idx];
            // grad_source_location[source_idx * 3 + 2] +=
            //     partial_simulate_record_idx_partial_zs * dL_dsimulate_record[idx];
            grad_source_p0[source_idx] += partial_simulate_record_idx_partial_p0 * dL_dsimulate_record[idx];
            grad_source_dx[source_idx] += partial_simulate_record_idx_partial_a0 * dL_dsimulate_record[idx];
        }
    }
}

// Function to call backward CUDA kernel
std::vector<torch::Tensor> simulate_cuda_backward(
    const torch::Tensor sensor_location, const torch::Tensor source_location, const torch::Tensor source_p0,
    const torch::Tensor source_dx, const torch::Tensor dL_dsimulate_record, const float dt, const size_t num_sensors,
    const size_t num_sources, const size_t num_times)
{
    auto grad_source_location = torch::zeros({static_cast<int64_t>(num_sources * 3)}, source_location.options());
    auto grad_source_p0 = torch::zeros({static_cast<int64_t>(num_sources)}, source_p0.options());
    auto grad_source_dx = torch::zeros({static_cast<int64_t>(num_sources)}, source_dx.options());

    const size_t thread_per_block = 256;
    const size_t num_blocks = (num_sources + thread_per_block - 1) / thread_per_block;

    simulate_cuda_backward_kernel<<<num_blocks, thread_per_block>>>(
        sensor_location.data_ptr<float>(), source_location.data_ptr<float>(), source_p0.data_ptr<float>(),
        source_dx.data_ptr<float>(), dL_dsimulate_record.data_ptr<float>(), dt, num_sensors, num_sources, num_times,
        grad_source_location.data_ptr<float>(), grad_source_p0.data_ptr<float>(), grad_source_dx.data_ptr<float>());
    cudaDeviceSynchronize();

    // catch error
    // cudaError_t error = cudaGetLastError();
    // printf("CUDA error: %s\n", cudaGetErrorString(error));

    return {grad_source_location, grad_source_p0, grad_source_dx};
}
