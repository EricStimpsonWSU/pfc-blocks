#include <fftw3.h>
#include <iostream>
#include <vector>
#include <complex>

int main() {
    const int N = 8; // Size of the FFT
    auto in = std::vector<double>(N);
    auto out = std::vector<double>(N);
    auto out_complex = std::vector<std::complex<double>>(N);
    auto in2d = std::vector<double>(N * N);
    auto out2d = std::vector<double>(N * N);
    auto out_complex2d = std::vector<std::complex<double>>(N * N);

    fftw_plan plan = NULL;
    fftw_plan plan_r2c = NULL;
    fftw_plan plan_r2c_2d = NULL;

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        in[i] = i + 1; // Example data
        for (int j = 0; j < N; ++j) {
            in2d[i * N + j] = (i + 1) + (j + 1) * 0.1;
        }
    }

    // Create FFT plan
    plan = fftw_plan_r2r_1d(N, in.data(), out.data(), FFTW_R2HC, FFTW_ESTIMATE);
    plan_r2c = fftw_plan_dft_r2c_1d(N, in.data(), reinterpret_cast<fftw_complex*>(out_complex.data()), FFTW_ESTIMATE);
    plan_r2c_2d = fftw_plan_dft_r2c_2d(N, N, in2d.data(), reinterpret_cast<fftw_complex*>(out_complex2d.data()), FFTW_ESTIMATE);
 
    // Execute FFT
    fftw_execute(plan);
    fftw_execute(plan_r2c);
    fftw_execute(plan_r2c_2d);

    // Print output
    std::cout << "FFT Output:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << out[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    fftw_destroy_plan(plan);
    fftw_cleanup();

    return 0;
}