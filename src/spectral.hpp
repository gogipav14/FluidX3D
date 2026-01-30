#pragma once

#include "defines.hpp"

#ifdef SPECTRAL_OPS

#include "opencl.hpp"

// VkFFT header - requires VkFFT to be in src/VkFFT/vkFFT.h
// For OpenCL backend, define VKFFT_BACKEND=3 before including
#define VKFFT_BACKEND 3 // OpenCL backend
#include "VkFFT/vkFFT.h"

class SpectralOps {
private:
	// Grid dimensions
	uint Nx = 0u, Ny = 0u, Nz = 0u;
	ulong N = 0ull;
	ulong N_complex = 0ull; // (Nx/2+1) * Ny * Nz for R2C output

	// VkFFT applications (plans)
	VkFFTApplication app_r2c; // Real-to-Complex forward FFT
	VkFFTApplication app_c2r; // Complex-to-Real inverse FFT
	bool plans_initialized = false;

	// OpenCL resources (from Device)
	cl_context cl_ctx = nullptr;
	cl_command_queue cl_queue_handle = nullptr;
	cl_device_id cl_dev = nullptr;

	// Device pointer for accessing command queue
	Device* device = nullptr;

	// Work buffers
	Memory<float>* buffer_real = nullptr;    // [N] temp buffer for real field
	Memory<float>* buffer_complex = nullptr; // [N_complex * 2] for complex k-space data
	Memory<float>* reduction_buffer = nullptr; // For parallel reduction (sum/mean)

#ifdef SPECTRAL_SUBGRID
	// Velocity FFT buffers (reuse buffer_complex for one component)
	Memory<float>* ux_hat = nullptr;  // [N_complex * 2] FFT of ux
	Memory<float>* uy_hat = nullptr;  // [N_complex * 2] FFT of uy
	// uz_hat uses buffer_complex
	Memory<float>* temp_complex = nullptr; // [N_complex * 2] temp for S_ij computation
	Memory<float>* S_sq_accum = nullptr;   // [N] accumulator for |S|^2 in physical space
	float Cs_delta_sq = SPECTRAL_CS_DELTA_SQ; // Smagorinsky (C_s * delta)^2
#endif

	// Pre-computed wavenumber arrays
	Memory<float>* kx = nullptr;       // [Nx/2+1] wavenumbers in x
	Memory<float>* ky = nullptr;       // [Ny] wavenumbers in y (FFT-ordered)
	Memory<float>* kz = nullptr;       // [Nz] wavenumbers in z (FFT-ordered)
	Memory<float>* k_mag_sq = nullptr; // [N_complex] pre-computed |k|^2

	// Spectral operation kernels
	Kernel kernel_helmholtz;         // Helmholtz smoother: 1/(1 + alpha*k^2)
	Kernel kernel_lowpass;           // Low-pass filter: exp(-k^2/(2*k_c^2))
	Kernel kernel_normalize;         // Normalize after IFFT: field /= N
	Kernel kernel_mass_correction;   // Subtract mean delta from field
	Kernel kernel_reduce_sum;        // Parallel sum reduction

	// Reduction helper
	float compute_field_sum(Memory<float>& field); // Returns sum of all elements in field

#ifdef SPECTRAL_SUBGRID
	Kernel kernel_extract_velocity;   // Extract velocity component from AoS
	Kernel kernel_compute_Sij;        // Compute strain component in k-space
	Kernel kernel_accumulate_S_sq;    // Square S_ij and accumulate to |S|^2
	Kernel kernel_compute_nu_t;       // Compute nu_t = (Cs*delta)^2 * sqrt(|S|^2)
	Kernel kernel_zero_field;         // Zero a field
#endif

#ifdef SPECTRAL_TEMPERATURE
	Kernel kernel_diffusion_etd;     // ETD diffusion: exp(-alpha*k^2*dt)
#endif

	// Configuration
	uint smooth_cadence = SPECTRAL_SMOOTH_EVERY;
	float helmholtz_alpha = SPECTRAL_HELMHOLTZ_ALPHA;

	// Internal methods
	void create_fft_plans();
	void destroy_fft_plans();
	void compute_wavenumbers();
	void allocate_buffers();
	void create_kernels();

public:
	SpectralOps() = default;
	SpectralOps(Device& device, const uint Nx, const uint Ny, const uint Nz);
	~SpectralOps();

	// Prevent copying (VkFFT plans are not copyable)
	SpectralOps(const SpectralOps&) = delete;
	SpectralOps& operator=(const SpectralOps&) = delete;

	// Allow move semantics
	SpectralOps(SpectralOps&& other) noexcept;
	SpectralOps& operator=(SpectralOps&& other) noexcept;

	// Core FFT operations (async, call finish_queue() after if needed)
	void enqueue_forward_r2c(Memory<float>& field_in);   // real field -> buffer_complex
	void enqueue_inverse_c2r(Memory<float>& field_out);  // buffer_complex -> real field

#ifdef SPECTRAL_SUBGRID
	// Helper FFT operations for SUBGRID (custom input/output buffers)
	void enqueue_forward_r2c_to(Memory<float>& field_in, Memory<float>& complex_out);
	void enqueue_inverse_c2r_from(Memory<float>& complex_in, Memory<float>& field_out);
#endif

	// High-level spectral operations for SURFACE
#ifdef SPECTRAL_SURFACE
	void enqueue_smooth_phi(Memory<float>& phi, const ulong timestep);
#endif

	// High-level spectral operations for SUBGRID
#ifdef SPECTRAL_SUBGRID
	void enqueue_compute_eddy_viscosity(Memory<float>& u, Memory<float>& nu_t);
#endif

	// High-level spectral operations for TEMPERATURE
#ifdef SPECTRAL_TEMPERATURE
	void enqueue_diffusion_step(Memory<float>& T, const float alpha_dt);
#endif

	// Utilities
	bool is_initialized() const { return plans_initialized; }
	ulong get_N_complex() const { return N_complex; }
	void set_smooth_cadence(const uint cadence) { smooth_cadence = cadence; }
	void set_helmholtz_alpha(const float alpha) { helmholtz_alpha = alpha; }
	uint get_smooth_cadence() const { return smooth_cadence; }
	float get_helmholtz_alpha() const { return helmholtz_alpha; }
#ifdef SPECTRAL_SUBGRID
	void set_Cs_delta_sq(const float val) { Cs_delta_sq = val; }
	float get_Cs_delta_sq() const { return Cs_delta_sq; }
#endif
	void finish_queue();

	// For device_defines() integration - returns OpenCL preprocessor defines
	string spectral_defines() const;
};

// Get stringified OpenCL C code for spectral kernels
string get_spectral_opencl_c_code();

#endif // SPECTRAL_OPS
