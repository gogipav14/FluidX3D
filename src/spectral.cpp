#include "spectral.hpp"

#ifdef SPECTRAL_OPS

#include <cmath>

// Constructor
SpectralOps::SpectralOps(Device& device, const uint Nx, const uint Ny, const uint Nz) {
	this->device = &device;
	this->Nx = Nx;
	this->Ny = Ny;
	this->Nz = Nz;
	this->N = (ulong)Nx * (ulong)Ny * (ulong)Nz;
	this->N_complex = (ulong)(Nx/2 + 1) * (ulong)Ny * (ulong)Nz;

	// Get OpenCL handles from Device
	cl_ctx = device.get_cl_context()();
	cl_queue_handle = device.get_cl_queue()();
	cl_dev = device.info.cl_device();

	// Allocate buffers and compute wavenumbers
	allocate_buffers();
	compute_wavenumbers();

	// Create VkFFT plans
	create_fft_plans();

	// Create spectral operation kernels
	create_kernels();

	print_info("SpectralOps initialized: " + to_string(Nx) + "x" + to_string(Ny) + "x" + to_string(Nz) +
	           " (N_complex=" + to_string(N_complex) + ")");
}

// Destructor
SpectralOps::~SpectralOps() {
	destroy_fft_plans();

	delete buffer_real;
	delete buffer_complex;
	delete reduction_buffer;
	delete kx;
	delete ky;
	delete kz;
	delete k_mag_sq;

#ifdef SPECTRAL_SUBGRID
	delete ux_hat;
	delete uy_hat;
	delete temp_complex;
	delete S_sq_accum;
#endif
}

// Move constructor
SpectralOps::SpectralOps(SpectralOps&& other) noexcept {
	*this = std::move(other);
}

// Move assignment
SpectralOps& SpectralOps::operator=(SpectralOps&& other) noexcept {
	if (this != &other) {
		destroy_fft_plans();
		delete buffer_real;
		delete buffer_complex;
		delete reduction_buffer;
		delete kx;
		delete ky;
		delete kz;
		delete k_mag_sq;
#ifdef SPECTRAL_SUBGRID
		delete ux_hat;
		delete uy_hat;
		delete temp_complex;
		delete S_sq_accum;
#endif

		device = other.device;
		Nx = other.Nx;
		Ny = other.Ny;
		Nz = other.Nz;
		N = other.N;
		N_complex = other.N_complex;
		cl_ctx = other.cl_ctx;
		cl_queue_handle = other.cl_queue_handle;
		cl_dev = other.cl_dev;
		app_r2c = other.app_r2c;
		app_c2r = other.app_c2r;
		plans_initialized = other.plans_initialized;
		buffer_real = other.buffer_real;
		buffer_complex = other.buffer_complex;
		reduction_buffer = other.reduction_buffer;
		kx = other.kx;
		ky = other.ky;
		kz = other.kz;
		k_mag_sq = other.k_mag_sq;
		smooth_cadence = other.smooth_cadence;
		helmholtz_alpha = other.helmholtz_alpha;

		// Kernels are copied by value
		kernel_helmholtz = other.kernel_helmholtz;
		kernel_lowpass = other.kernel_lowpass;
		kernel_normalize = other.kernel_normalize;
		kernel_mass_correction = other.kernel_mass_correction;
		kernel_reduce_sum = other.kernel_reduce_sum;

#ifdef SPECTRAL_SUBGRID
		ux_hat = other.ux_hat;
		uy_hat = other.uy_hat;
		temp_complex = other.temp_complex;
		S_sq_accum = other.S_sq_accum;
		Cs_delta_sq = other.Cs_delta_sq;
		kernel_extract_velocity = other.kernel_extract_velocity;
		kernel_compute_Sij = other.kernel_compute_Sij;
		kernel_accumulate_S_sq = other.kernel_accumulate_S_sq;
		kernel_compute_nu_t = other.kernel_compute_nu_t;
		kernel_zero_field = other.kernel_zero_field;
		other.ux_hat = nullptr;
		other.uy_hat = nullptr;
		other.temp_complex = nullptr;
		other.S_sq_accum = nullptr;
#endif

#ifdef SPECTRAL_TEMPERATURE
		kernel_diffusion_etd = other.kernel_diffusion_etd;
#endif

		// Null out other's pointers to prevent double-free
		other.buffer_real = nullptr;
		other.buffer_complex = nullptr;
		other.reduction_buffer = nullptr;
		other.kx = nullptr;
		other.ky = nullptr;
		other.kz = nullptr;
		other.k_mag_sq = nullptr;
		other.plans_initialized = false;
	}
	return *this;
}

void SpectralOps::allocate_buffers() {
	// Real buffer for temporary storage
	buffer_real = new Memory<float>(*device, N, 1u, false, true, 0.0f, false); // device only

	// Complex buffer: (Nx/2+1) * Ny * Nz complex values = 2 floats each
	buffer_complex = new Memory<float>(*device, N_complex, 2u, false, true, 0.0f, false); // device only

	// Reduction buffer for parallel sum (256 work groups max)
	const ulong reduction_size = 256ull;
	reduction_buffer = new Memory<float>(*device, reduction_size, 1u, true, true, 0.0f, false);

	// Wavenumber arrays
	kx = new Memory<float>(*device, Nx/2 + 1, 1u, true, true, 0.0f, false);
	ky = new Memory<float>(*device, Ny, 1u, true, true, 0.0f, false);
	kz = new Memory<float>(*device, Nz, 1u, true, true, 0.0f, false);

	// Pre-computed |k|^2 for all k-space points
	k_mag_sq = new Memory<float>(*device, N_complex, 1u, true, true, 0.0f, false);

#ifdef SPECTRAL_SUBGRID
	// Velocity FFT buffers: store all 3 components in k-space simultaneously
	ux_hat = new Memory<float>(*device, N_complex, 2u, false, true, 0.0f, false);
	uy_hat = new Memory<float>(*device, N_complex, 2u, false, true, 0.0f, false);
	// uz_hat reuses buffer_complex

	// Temp complex buffer for S_ij computation
	temp_complex = new Memory<float>(*device, N_complex, 2u, false, true, 0.0f, false);

	// Accumulator for |S|^2 in physical space
	S_sq_accum = new Memory<float>(*device, N, 1u, false, true, 0.0f, false);
#endif
}

void SpectralOps::compute_wavenumbers() {
	const float two_pi_Nx = 6.283185307f / (float)Nx;
	const float two_pi_Ny = 6.283185307f / (float)Ny;
	const float two_pi_Nz = 6.283185307f / (float)Nz;

	// kx: [0, 1, 2, ..., Nx/2] * (2*pi/Nx)
	for (uint i = 0; i <= Nx/2; i++) {
		kx->data()[i] = (float)i * two_pi_Nx;
	}

	// ky: FFT-ordered [0, 1, ..., Ny/2, -Ny/2+1, ..., -1] * (2*pi/Ny)
	for (uint j = 0; j < Ny; j++) {
		int jj = (j <= Ny/2) ? (int)j : (int)j - (int)Ny;
		ky->data()[j] = (float)jj * two_pi_Ny;
	}

	// kz: FFT-ordered [0, 1, ..., Nz/2, -Nz/2+1, ..., -1] * (2*pi/Nz)
	for (uint k = 0; k < Nz; k++) {
		int kk = (k <= Nz/2) ? (int)k : (int)k - (int)Nz;
		kz->data()[k] = (float)kk * two_pi_Nz;
	}

	// Pre-compute |k|^2 for all k-space points
	const uint Nx_half = Nx/2 + 1;
	for (uint k = 0; k < Nz; k++) {
		for (uint j = 0; j < Ny; j++) {
			for (uint i = 0; i < Nx_half; i++) {
				const ulong idx = (ulong)i + (ulong)j * Nx_half + (ulong)k * Nx_half * Ny;
				const float kx_val = kx->data()[i];
				const float ky_val = ky->data()[j];
				const float kz_val = kz->data()[k];
				k_mag_sq->data()[idx] = kx_val*kx_val + ky_val*ky_val + kz_val*kz_val;
			}
		}
	}

	// Transfer to device
	kx->write_to_device();
	ky->write_to_device();
	kz->write_to_device();
	k_mag_sq->write_to_device();
}

void SpectralOps::create_fft_plans() {
	VkFFTResult result;
	VkFFTConfiguration config = {};

	// Common configuration
	config.FFTdim = 3;
	config.size[0] = Nx;
	config.size[1] = Ny;
	config.size[2] = Nz;
	config.performR2C = 1; // Real-to-Complex

	// OpenCL specific
	config.device = &cl_dev;
	config.context = &cl_ctx;

	// Buffer configuration for R2C
	uint64_t buffer_size_complex = N_complex * 2 * sizeof(float);
	config.bufferSize = &buffer_size_complex;

	cl_mem complex_buf = buffer_complex->get_cl_buffer()();
	config.buffer = &complex_buf;

	// Create forward R2C plan
	config.inverseReturnToInputBuffer = 0;
	result = initializeVkFFT(&app_r2c, config);
	if (result != VKFFT_SUCCESS) {
		print_error("VkFFT R2C plan creation failed with error " + to_string((int)result));
		return;
	}

	// Create inverse C2R plan
	config.inverseReturnToInputBuffer = 1; // output to input buffer for C2R
	result = initializeVkFFT(&app_c2r, config);
	if (result != VKFFT_SUCCESS) {
		print_error("VkFFT C2R plan creation failed with error " + to_string((int)result));
		deleteVkFFT(&app_r2c);
		return;
	}

	plans_initialized = true;
}

void SpectralOps::destroy_fft_plans() {
	if (plans_initialized) {
		deleteVkFFT(&app_r2c);
		deleteVkFFT(&app_c2r);
		plans_initialized = false;
	}
}

void SpectralOps::create_kernels() {
	// Helmholtz smoother kernel
	kernel_helmholtz = Kernel(*device, N_complex, "spectral_helmholtz",
	                          *buffer_complex, *k_mag_sq, helmholtz_alpha);

	// Low-pass filter kernel
	kernel_lowpass = Kernel(*device, N_complex, "spectral_lowpass",
	                        *buffer_complex, *k_mag_sq, 1.0f); // cutoff_k_sq placeholder

	// Normalize kernel (divide by N after IFFT)
	kernel_normalize = Kernel(*device, N, "spectral_normalize",
	                          *buffer_real, 1.0f / (float)N);

	// Mass correction kernel
	kernel_mass_correction = Kernel(*device, N, "spectral_mass_correction",
	                                *buffer_real, 0.0f); // delta_mean placeholder

	// Reduction kernel for computing sum
	kernel_reduce_sum = Kernel(*device, 256ull, "spectral_reduce_sum",
	                           *buffer_real, *reduction_buffer, N);

#ifdef SPECTRAL_SUBGRID
	// Extract velocity component: u[n*3 + dim] -> buffer_real[n]
	kernel_extract_velocity = Kernel(*device, N, "spectral_extract_velocity",
	                                 *buffer_real, *buffer_real, 0u); // u, buffer_real, dimension placeholder

	// Compute S_ij in k-space from velocity FFTs
	// component: 0=Sxx, 1=Syy, 2=Szz, 3=Sxy, 4=Sxz, 5=Syz
	kernel_compute_Sij = Kernel(*device, N_complex, "spectral_compute_Sij",
	                            *ux_hat, *uy_hat, *buffer_complex, // ux_hat, uy_hat, uz_hat
	                            *temp_complex, // output S_ij in k-space
	                            *kx, *ky, *kz, 0u); // wavenumbers, component

	// Accumulate |S_ij|^2 to accumulator (after IFFT)
	// factor is 1.0 for diagonal (Sxx, Syy, Szz) and 2.0 for off-diagonal (Sxy, Sxz, Syz)
	kernel_accumulate_S_sq = Kernel(*device, N, "spectral_accumulate_S_sq",
	                                *buffer_real, *S_sq_accum, 1.0f); // S_ij, accumulator, factor

	// Compute nu_t = (Cs*delta)^2 * sqrt(2 * |S|^2)
	kernel_compute_nu_t = Kernel(*device, N, "spectral_compute_nu_t",
	                             *S_sq_accum, *buffer_real, Cs_delta_sq); // S_sq_accum, nu_t, Cs_delta_sq

	// Zero a field
	kernel_zero_field = Kernel(*device, N, "spectral_zero_field", *S_sq_accum);
#endif

#ifdef SPECTRAL_TEMPERATURE
	// ETD diffusion kernel
	kernel_diffusion_etd = Kernel(*device, N_complex, "spectral_diffusion_etd",
	                              *buffer_complex, *k_mag_sq, 0.0f); // alpha_dt placeholder
#endif
}

void SpectralOps::enqueue_forward_r2c(Memory<float>& field_in) {
	if (!plans_initialized) return;

	VkFFTLaunchParams params = {};
	params.commandQueue = &cl_queue_handle;

	cl_mem input_buf = field_in.get_cl_buffer()();
	cl_mem output_buf = buffer_complex->get_cl_buffer()();
	params.inputBuffer = &input_buf;
	params.buffer = &output_buf;

	VkFFTResult result = VkFFTAppend(&app_r2c, -1, &params); // -1 = forward
	if (result != VKFFT_SUCCESS) {
		print_error("VkFFT forward R2C failed with error " + to_string((int)result));
	}
}

void SpectralOps::enqueue_inverse_c2r(Memory<float>& field_out) {
	if (!plans_initialized) return;

	VkFFTLaunchParams params = {};
	params.commandQueue = &cl_queue_handle;

	cl_mem input_buf = buffer_complex->get_cl_buffer()();
	cl_mem output_buf = field_out.get_cl_buffer()();
	params.buffer = &input_buf;
	params.outputBuffer = &output_buf;

	VkFFTResult result = VkFFTAppend(&app_c2r, 1, &params); // 1 = inverse
	if (result != VKFFT_SUCCESS) {
		print_error("VkFFT inverse C2R failed with error " + to_string((int)result));
	}

	// Normalize: VkFFT doesn't normalize by default
	kernel_normalize.set_parameters(0, field_out, 1.0f / (float)N).enqueue_run();
}

float SpectralOps::compute_field_sum(Memory<float>& field) {
	// Run parallel reduction kernel
	kernel_reduce_sum.set_parameters(0, field, *reduction_buffer, N).enqueue_run();

	// Read back partial sums (256 values)
	reduction_buffer->read_from_device();

	// Sum on CPU (small array, fast)
	float total = 0.0f;
	const ulong num_groups = 256ull;
	for (ulong i = 0; i < num_groups; i++) {
		total += reduction_buffer->data()[i];
	}
	return total;
}

#ifdef SPECTRAL_SURFACE
void SpectralOps::enqueue_smooth_phi(Memory<float>& phi, const ulong timestep) {
	// Only smooth every smooth_cadence steps
	if (timestep % smooth_cadence != 0) return;
	if (!plans_initialized) return;

	// Step 1: Compute sum(phi) before smoothing (for mass conservation)
	const float sum_before = compute_field_sum(phi);
	const float mean_before = sum_before / (float)N;

	// Step 2: Forward FFT: phi -> phi_hat (in buffer_complex)
	enqueue_forward_r2c(phi);

	// Step 3: Apply Helmholtz filter in k-space: phi_hat *= 1/(1 + alpha*k^2)
	// Note: H(0) = 1, so mean is preserved in exact arithmetic
	// But numerical errors can accumulate, so we correct explicitly
	kernel_helmholtz.set_parameters(2, helmholtz_alpha).enqueue_run();

	// Step 4: Inverse FFT: phi_hat -> phi
	enqueue_inverse_c2r(phi);

	// Step 5: Compute sum(phi) after smoothing
	const float sum_after = compute_field_sum(phi);
	const float mean_after = sum_after / (float)N;

	// Step 6: Apply mass correction: phi -= (mean_after - mean_before)
	// This ensures sum(phi) is exactly preserved
	const float delta_mean = mean_after - mean_before;
	if (fabs(delta_mean) > 1e-10f) { // Only correct if there's a measurable difference
		kernel_mass_correction.set_parameters(0, phi, delta_mean).enqueue_run();
	}
}
#endif

#ifdef SPECTRAL_SUBGRID
void SpectralOps::enqueue_compute_eddy_viscosity(Memory<float>& u, Memory<float>& nu_t) {
	if (!plans_initialized) return;

	// Step 1: Zero the |S|^2 accumulator
	kernel_zero_field.set_parameters(0, *S_sq_accum).enqueue_run();

	// Step 2: FFT all 3 velocity components
	// Extract ux (dimension 0) -> buffer_real, then FFT -> ux_hat
	kernel_extract_velocity.set_parameters(0, u, *buffer_real, 0u).enqueue_run();
	enqueue_forward_r2c_to(*buffer_real, *ux_hat);

	// Extract uy (dimension 1) -> buffer_real, then FFT -> uy_hat
	kernel_extract_velocity.set_parameters(0, u, *buffer_real, 1u).enqueue_run();
	enqueue_forward_r2c_to(*buffer_real, *uy_hat);

	// Extract uz (dimension 2) -> buffer_real, then FFT -> buffer_complex (uz_hat)
	kernel_extract_velocity.set_parameters(0, u, *buffer_real, 2u).enqueue_run();
	enqueue_forward_r2c(*buffer_real); // Result in buffer_complex

	// Step 3: Compute each strain component, IFFT, and accumulate |S|^2
	// Strain components: 0=Sxx, 1=Syy, 2=Szz (diagonal, factor=1)
	//                    3=Sxy, 4=Sxz, 5=Syz (off-diagonal, factor=2)

	// Sxx = dux/dx
	kernel_compute_Sij.set_parameters(0, *ux_hat, *uy_hat, *buffer_complex, *temp_complex,
	                                  *kx, *ky, *kz, 0u).enqueue_run();
	enqueue_inverse_c2r_from(*temp_complex, *buffer_real);
	kernel_accumulate_S_sq.set_parameters(0, *buffer_real, *S_sq_accum, 1.0f).enqueue_run();

	// Syy = duy/dy
	kernel_compute_Sij.set_parameters(7, 1u).enqueue_run(); // Just change component
	enqueue_inverse_c2r_from(*temp_complex, *buffer_real);
	kernel_accumulate_S_sq.enqueue_run();

	// Szz = duz/dz
	kernel_compute_Sij.set_parameters(7, 2u).enqueue_run();
	enqueue_inverse_c2r_from(*temp_complex, *buffer_real);
	kernel_accumulate_S_sq.enqueue_run();

	// Sxy = 0.5*(dux/dy + duy/dx), off-diagonal: factor=2
	kernel_compute_Sij.set_parameters(7, 3u).enqueue_run();
	enqueue_inverse_c2r_from(*temp_complex, *buffer_real);
	kernel_accumulate_S_sq.set_parameters(2, 2.0f).enqueue_run();

	// Sxz = 0.5*(dux/dz + duz/dx)
	kernel_compute_Sij.set_parameters(7, 4u).enqueue_run();
	enqueue_inverse_c2r_from(*temp_complex, *buffer_real);
	kernel_accumulate_S_sq.enqueue_run();

	// Syz = 0.5*(duy/dz + duz/dy)
	kernel_compute_Sij.set_parameters(7, 5u).enqueue_run();
	enqueue_inverse_c2r_from(*temp_complex, *buffer_real);
	kernel_accumulate_S_sq.enqueue_run();

	// Step 4: Compute nu_t = (Cs*delta)^2 * sqrt(2 * |S|^2)
	kernel_compute_nu_t.set_parameters(0, *S_sq_accum, nu_t, Cs_delta_sq).enqueue_run();
}

// Helper: forward FFT to specific output buffer
void SpectralOps::enqueue_forward_r2c_to(Memory<float>& field_in, Memory<float>& complex_out) {
	if (!plans_initialized) return;

	VkFFTLaunchParams params = {};
	params.commandQueue = &cl_queue_handle;

	cl_mem input_buf = field_in.get_cl_buffer()();
	cl_mem output_buf = complex_out.get_cl_buffer()();
	params.inputBuffer = &input_buf;
	params.buffer = &output_buf;

	VkFFTResult result = VkFFTAppend(&app_r2c, -1, &params);
	if (result != VKFFT_SUCCESS) {
		print_error("VkFFT forward R2C failed with error " + to_string((int)result));
	}
}

// Helper: inverse FFT from specific input buffer
void SpectralOps::enqueue_inverse_c2r_from(Memory<float>& complex_in, Memory<float>& field_out) {
	if (!plans_initialized) return;

	VkFFTLaunchParams params = {};
	params.commandQueue = &cl_queue_handle;

	cl_mem input_buf = complex_in.get_cl_buffer()();
	cl_mem output_buf = field_out.get_cl_buffer()();
	params.buffer = &input_buf;
	params.outputBuffer = &output_buf;

	VkFFTResult result = VkFFTAppend(&app_c2r, 1, &params);
	if (result != VKFFT_SUCCESS) {
		print_error("VkFFT inverse C2R failed with error " + to_string((int)result));
	}

	// Normalize
	kernel_normalize.set_parameters(0, field_out, 1.0f / (float)N).enqueue_run();
}
#endif

#ifdef SPECTRAL_TEMPERATURE
void SpectralOps::enqueue_diffusion_step(Memory<float>& T, const float alpha_dt) {
	if (!plans_initialized) return;

	// Forward FFT: T -> T_hat
	enqueue_forward_r2c(T);

	// Apply ETD diffusion: T_hat *= exp(-alpha*k^2*dt)
	kernel_diffusion_etd.set_parameters(2, alpha_dt).enqueue_run();

	// Inverse FFT: T_hat -> T
	enqueue_inverse_c2r(T);
}
#endif

void SpectralOps::finish_queue() {
	device->finish_queue();
}

string SpectralOps::spectral_defines() const {
	return
		"\n	#define def_spectral_Nx " + to_string(Nx) + "u"
		"\n	#define def_spectral_Ny " + to_string(Ny) + "u"
		"\n	#define def_spectral_Nz " + to_string(Nz) + "u"
		"\n	#define def_spectral_N " + to_string(N) + "ul"
		"\n	#define def_spectral_N_complex " + to_string(N_complex) + "ul"
		"\n	#define def_spectral_Nx_half " + to_string(Nx/2 + 1) + "u"
	;
}

// Stringified OpenCL C code for spectral kernels
string get_spectral_opencl_c_code() { return R(

// Helmholtz smoother: H(k) = 1/(1 + alpha*|k|^2)
// Preserves k=0 mode (mean) exactly since H(0) = 1
kernel void spectral_helmholtz(
	global float* complex_data,       // [N_complex * 2] interleaved Re/Im
	const global float* k_mag_sq,     // [N_complex] pre-computed |k|^2
	const float alpha                 // smoothing parameter
) {
	const ulong idx = get_global_id(0);
	if (idx >= def_spectral_N_complex) return;

	const float k2 = k_mag_sq[idx];
	const float filter = 1.0f / (1.0f + alpha * k2);

	complex_data[2*idx    ] *= filter; // Real part
	complex_data[2*idx + 1] *= filter; // Imaginary part
}

// Low-pass filter: H(k) = exp(-|k|^2 / (2*k_cutoff^2))
kernel void spectral_lowpass(
	global float* complex_data,
	const global float* k_mag_sq,
	const float cutoff_k_sq           // k_cutoff^2
) {
	const ulong idx = get_global_id(0);
	if (idx >= def_spectral_N_complex) return;

	const float k2 = k_mag_sq[idx];
	const float filter = exp(-k2 / (2.0f * cutoff_k_sq));

	complex_data[2*idx    ] *= filter;
	complex_data[2*idx + 1] *= filter;
}

// Normalize field after inverse FFT (divide by N)
kernel void spectral_normalize(
	global float* field,
	const float inv_N                 // 1/N
) {
	const ulong idx = get_global_id(0);
	if (idx >= def_spectral_N) return;

	field[idx] *= inv_N;
}

// Mass correction: subtract delta_mean from all cells
kernel void spectral_mass_correction(
	global float* field,
	const float delta_mean            // mean(new) - mean(old)
) {
	const ulong idx = get_global_id(0);
	if (idx >= def_spectral_N) return;

	field[idx] -= delta_mean;
}

// Parallel sum reduction: computes partial sums for 256 work groups
// Each work group reduces a chunk of the input to a single value
kernel void spectral_reduce_sum(
	const global float* field,        // Input field to sum
	global float* partial_sums,       // Output: 256 partial sums
	const ulong field_size            // Size of input field
) {
	const uint group_id = get_group_id(0);
	const uint local_id = get_local_id(0);
	const uint local_size = get_local_size(0);
	const uint num_groups = get_num_groups(0);

	// Each work group handles a chunk of the input
	const ulong chunk_size = (field_size + num_groups - 1) / num_groups;
	const ulong start = (ulong)group_id * chunk_size;
	const ulong end = min(start + chunk_size, field_size);

	// Local memory for reduction within work group
	local float local_sum[256];

	// Each work item accumulates its portion
	float sum = 0.0f;
	for (ulong i = start + local_id; i < end; i += local_size) {
		sum += field[i];
	}
	local_sum[local_id] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Tree reduction within work group
	for (uint stride = local_size / 2; stride > 0; stride /= 2) {
		if (local_id < stride) {
			local_sum[local_id] += local_sum[local_id + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// First work item writes result
	if (local_id == 0) {
		partial_sums[group_id] = local_sum[0];
	}
}

#ifdef SPECTRAL_SUBGRID
// Extract one velocity component from AoS format: u[n*3 + dim] -> out[n]
kernel void spectral_extract_velocity(
	const global float* u,            // [N * 3] velocity in AoS format
	global float* out,                // [N] extracted component
	const uint dimension              // 0=x, 1=y, 2=z
) {
	const ulong n = get_global_id(0);
	if (n >= def_spectral_N) return;
	out[n] = u[n * 3ul + (ulong)dimension];
}

// Zero a field
kernel void spectral_zero_field(global float* field) {
	const ulong idx = get_global_id(0);
	if (idx >= def_spectral_N) return;
	field[idx] = 0.0f;
}

// Compute strain component S_ij in k-space
// component: 0=Sxx, 1=Syy, 2=Szz, 3=Sxy, 4=Sxz, 5=Syz
kernel void spectral_compute_Sij(
	const global float* ux_hat,       // [N_complex * 2] velocity x in k-space
	const global float* uy_hat,       // [N_complex * 2] velocity y in k-space
	const global float* uz_hat,       // [N_complex * 2] velocity z in k-space
	global float* Sij_hat,            // [N_complex * 2] output: S_ij in k-space
	const global float* kx,           // [Nx/2+1] wavenumbers
	const global float* ky,           // [Ny] wavenumbers (FFT-ordered)
	const global float* kz,           // [Nz] wavenumbers (FFT-ordered)
	const uint component              // which strain component
) {
	const ulong idx = get_global_id(0);
	if (idx >= def_spectral_N_complex) return;

	// Compute (i, j, k) indices from linear index
	const uint i = idx % def_spectral_Nx_half;
	const uint j = (idx / def_spectral_Nx_half) % def_spectral_Ny;
	const uint k = idx / (def_spectral_Nx_half * def_spectral_Ny);

	const float kx_val = kx[i];
	const float ky_val = ky[j];
	const float kz_val = kz[k];

	// Load velocity components
	const float ux_re = ux_hat[2*idx], ux_im = ux_hat[2*idx + 1];
	const float uy_re = uy_hat[2*idx], uy_im = uy_hat[2*idx + 1];
	const float uz_re = uz_hat[2*idx], uz_im = uz_hat[2*idx + 1];

	// Spectral derivative: d/dx -> i*kx multiplies complex number
	// (a + bi) * (i*k) = -b*k + a*k*i
	float Sij_re = 0.0f, Sij_im = 0.0f;

	switch (component) {
		case 0: // Sxx = dux/dx
			Sij_re = -ux_im * kx_val;
			Sij_im =  ux_re * kx_val;
			break;
		case 1: // Syy = duy/dy
			Sij_re = -uy_im * ky_val;
			Sij_im =  uy_re * ky_val;
			break;
		case 2: // Szz = duz/dz
			Sij_re = -uz_im * kz_val;
			Sij_im =  uz_re * kz_val;
			break;
		case 3: // Sxy = 0.5*(dux/dy + duy/dx)
			Sij_re = 0.5f * (-ux_im * ky_val - uy_im * kx_val);
			Sij_im = 0.5f * ( ux_re * ky_val + uy_re * kx_val);
			break;
		case 4: // Sxz = 0.5*(dux/dz + duz/dx)
			Sij_re = 0.5f * (-ux_im * kz_val - uz_im * kx_val);
			Sij_im = 0.5f * ( ux_re * kz_val + uz_re * kx_val);
			break;
		case 5: // Syz = 0.5*(duy/dz + duz/dy)
			Sij_re = 0.5f * (-uy_im * kz_val - uz_im * ky_val);
			Sij_im = 0.5f * ( uy_re * kz_val + uz_re * ky_val);
			break;
	}

	Sij_hat[2*idx    ] = Sij_re;
	Sij_hat[2*idx + 1] = Sij_im;
}

// Accumulate |S_ij|^2 to accumulator: accum += factor * S_ij^2
kernel void spectral_accumulate_S_sq(
	const global float* Sij,          // [N] S_ij component in physical space
	global float* accum,              // [N] accumulator for |S|^2
	const float factor                // 1.0 for diagonal, 2.0 for off-diagonal
) {
	const ulong idx = get_global_id(0);
	if (idx >= def_spectral_N) return;
	const float s = Sij[idx];
	accum[idx] += factor * s * s;
}

// Compute eddy viscosity: nu_t = Cs_delta_sq * sqrt(2 * |S|^2)
kernel void spectral_compute_nu_t(
	const global float* S_sq_accum,   // [N] accumulated |S|^2
	global float* nu_t,               // [N] output eddy viscosity
	const float Cs_delta_sq           // (C_s * delta)^2
) {
	const ulong idx = get_global_id(0);
	if (idx >= def_spectral_N) return;
	// |S| = sqrt(2 * S_ij * S_ij), and we accumulated S_ij^2 with proper factors
	// So |S|^2 = 2 * sum(factor_ij * S_ij^2) where factor=1 for diag, 2 for off-diag
	// But we already included the factor 2 in accumulation formula
	// Actually, |S|^2 = 2*(Sxx^2 + Syy^2 + Szz^2 + 2*Sxy^2 + 2*Sxz^2 + 2*Syz^2)
	// We accumulated with factor=1 for diag and factor=2 for off-diag
	// So accum = Sxx^2 + Syy^2 + Szz^2 + 2*Sxy^2 + 2*Sxz^2 + 2*Syz^2
	// |S| = sqrt(2 * accum)
	const float S_mag = sqrt(2.0f * S_sq_accum[idx]);
	nu_t[idx] = Cs_delta_sq * S_mag;
}
#endif // SPECTRAL_SUBGRID

#ifdef SPECTRAL_TEMPERATURE
// ETD diffusion: T_hat *= exp(-alpha * |k|^2 * dt)
// This is the exact solution to dT/dt = alpha * laplacian(T)
kernel void spectral_diffusion_etd(
	global float* T_hat,
	const global float* k_mag_sq,
	const float alpha_dt              // alpha * dt
) {
	const ulong idx = get_global_id(0);
	if (idx >= def_spectral_N_complex) return;

	const float k2 = k_mag_sq[idx];
	const float decay = exp(-alpha_dt * k2);

	T_hat[2*idx    ] *= decay;
	T_hat[2*idx + 1] *= decay;
}
#endif // SPECTRAL_TEMPERATURE

)"+R(
// Additional spectral utility kernels can be added here
);}

#endif // SPECTRAL_OPS
