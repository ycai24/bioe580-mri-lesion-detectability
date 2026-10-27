"""BIOE 580 MRI lesion detectability -- Step 1: simulate k-space noise + Cartesian undersampling and save reconstructions.

Converted from the original Jupyter notebook. Cells are separated by
`# %%` markers, so the file can be run top-to-bottom as a script or
cell-by-cell in VS Code / Spyder.

The committed notebook mixed two versions of this step: its first cell had
been rewritten, but the later cells still called an older API and relied on
variables from a previous kernel session. Here the later cells use the
helpers defined in the first cell, and the generation cases cover every
reconstruction that c2_get_patch_datasets.py extracts patches from.
"""


# %%
import builtins
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from skimage.metrics import structural_similarity as ssim

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'MRI'
MASKS_DIR = PROJECT_ROOT / 'masks'

images_filepath = DATA_DIR / 'HCP_selected_ones2.npy'
with builtins.open(images_filepath, 'rb') as f:
    images = np.load(f, allow_pickle=True).astype(np.float32)
n_case_images = min(11_000, images.shape[0])
images = images[:n_case_images]
print('Original images shape:', images.shape)

images_lesion_indices_filepath = DATA_DIR / 'images_lesion_indices.bin'
with builtins.open(images_lesion_indices_filepath, 'rb') as f:
    lesion_indices_all = np.fromfile(f, dtype=np.int16).reshape(-1, 2)
images_lesion_indices = lesion_indices_all[:n_case_images]
print('Lesion-center indices shape:', images_lesion_indices.shape)

mask_2x_filepath = MASKS_DIR / 'mask_2x_cartesian_for_260x311.npy'
with builtins.open(mask_2x_filepath, 'rb') as f:
    mask_2x = np.load(f, allow_pickle=True)
print('x2 mask shape:', mask_2x.shape)

mask_4x_filepath = MASKS_DIR / 'mask_4x_cartesian_for_260x311.npy'
with builtins.open(mask_4x_filepath, 'rb') as f:
    mask_4x = np.load(f, allow_pickle=True)
print('x4 mask shape:', mask_4x.shape)

reference_lesion_intensity = 0.10
reference_noise_level = 0.10
reference_undersampling_factor = 2
lesion_intensity_cases = [0.05, 0.10, 0.20]
# Union of the noise levels that c2/c3 request. 0.005 is not listed: the `.2f`
# filenames used throughout the pipeline map it onto the 0.01 file anyway.
noise_levels = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
undersampling_factors = [1, 2, 4]
save_reconstructed_images = True
save_iq_metrics = True
n_batch_images = 500
random_seed = 0
preview_case_index = 0
preview_source_mode = 'lesion'
preview_noise_level = 0.00
preview_undersampling_factor = 1

sampling_masks = {
    1: np.ones(images.shape[1:], dtype=np.float32),
    2: mask_2x.astype(np.float32),
    4: mask_4x.astype(np.float32),
}

def build_source_tag(source_mode, lesion_intensity):
    if source_mode == 'original':
        return 'original'
    if lesion_intensity is None:
        raise ValueError('lesion_intensity is required when source_mode is lesion.')
    return f'lesion_{lesion_intensity:.2f}'

def build_case_label(source_mode, lesion_intensity, noise_level, undersampling_factor):
    source_tag = build_source_tag(source_mode, lesion_intensity)
    return (
        f'source={source_tag}, noise={noise_level:.2f}, '
        f'undersampling={undersampling_factor}x'
    )

def build_lesioned_volume_filepath(lesion_intensity):
    if np.isclose(lesion_intensity, reference_lesion_intensity):
        return DATA_DIR / 'images_lesioned.bin'
    return DATA_DIR / f'images_lesioned_intensity_{lesion_intensity:.2f}.bin'

def build_reconstructed_filepath(source_mode, lesion_intensity, noise_level, undersampling_factor):
    source_tag = build_source_tag(source_mode, lesion_intensity)
    return DATA_DIR / (
        f'images_reconstructed_source_{source_tag}'
        f'_noise_{noise_level:.2f}_undersample_{undersampling_factor}.bin'
    )

def build_iq_metrics_filepath(source_mode, lesion_intensity, noise_level, undersampling_factor):
    source_tag = build_source_tag(source_mode, lesion_intensity)
    return DATA_DIR / (
        f'iq_metrics_source_{source_tag}'
        f'_noise_{noise_level:.2f}_undersample_{undersampling_factor}.npz'
    )

def load_raw_volume(filepath, shape):
    # Memory-mapped so that single-image lookups don't read the whole volume.
    n_values = filepath.stat().st_size // np.dtype(np.float32).itemsize
    expected_size = int(np.prod(shape))
    if n_values < expected_size:
        raise ValueError(f'Volume {filepath} is smaller than expected for shape {shape}.')
    return np.memmap(filepath, dtype=np.float32, mode='r', shape=shape)

def load_source_images(source_mode, lesion_intensity):
    if source_mode == 'original':
        return images
    filepath = build_lesioned_volume_filepath(lesion_intensity)
    return load_raw_volume(filepath, images.shape)

def build_source_image(case_index, source_mode, lesion_intensity):
    if source_mode == 'original':
        return images[case_index]
    source_images = load_source_images(source_mode, lesion_intensity)
    return np.asarray(source_images[case_index], dtype=np.float32)

def noise_level_to_raw_kspace_component_sigma(noise_level):
    return float(noise_level * np.sqrt(np.prod(images.shape[1:]) / 2.0))

def reconstruct_from_kspace(source_image, noise_level, undersampling_factor, rng):
    shifted_kspace = fftshift(fftn(source_image))
    raw_sigma = noise_level_to_raw_kspace_component_sigma(noise_level)
    if raw_sigma > 0:
        real_noise = rng.normal(loc=0.0, scale=raw_sigma, size=shifted_kspace.shape)
        imag_noise = rng.normal(loc=0.0, scale=raw_sigma, size=shifted_kspace.shape)
        shifted_kspace = shifted_kspace + (real_noise + 1j * imag_noise)
    sampled_shifted_kspace = shifted_kspace * sampling_masks[undersampling_factor]
    reconstructed_image = np.abs(ifftn(ifftshift(sampled_shifted_kspace))).astype(np.float32)
    return reconstructed_image, sampled_shifted_kspace

def compute_image_quality_values(reference_images, reconstructed_images):
    mse_values = ((reference_images - reconstructed_images) ** 2).mean(axis=(1, 2))
    ssim_values = np.array([
        ssim(
            reference_images[idx],
            reconstructed_images[idx],
            data_range=float(reference_images[idx].max() - reference_images[idx].min() + 1e-8),
        )
        for idx in range(reference_images.shape[0])
    ], dtype=np.float32)
    return mse_values, ssim_values

def summarize_image_quality(mse_values, ssim_values):
    return {
        'mse_mean': float(mse_values.mean()),
        'mse_std': float(mse_values.std()),
        'ssim_mean': float(ssim_values.mean()),
        'ssim_std': float(ssim_values.std()),
        'n_images': int(mse_values.shape[0]),
    }

def compute_image_quality_metrics(reference_images, reconstructed_images):
    return summarize_image_quality(*compute_image_quality_values(reference_images, reconstructed_images))

def build_generation_cases():
    cases = []
    for undersampling_factor in undersampling_factors:
        cases.append({
            'source_mode': 'original',
            'lesion_intensity': None,
            'noise_level': 0.00,
            'undersampling_factor': undersampling_factor,
        })
    for noise_level in noise_levels:
        cases.append({
            'source_mode': 'lesion',
            'lesion_intensity': reference_lesion_intensity,
            'noise_level': noise_level,
            'undersampling_factor': reference_undersampling_factor,
        })
        # Signal-absent backgrounds at the same acquisition settings (used by c2).
        cases.append({
            'source_mode': 'original',
            'lesion_intensity': None,
            'noise_level': noise_level,
            'undersampling_factor': reference_undersampling_factor,
        })
    for lesion_intensity in lesion_intensity_cases:
        if np.isclose(lesion_intensity, reference_lesion_intensity):
            continue
        cases.append({
            'source_mode': 'lesion',
            'lesion_intensity': lesion_intensity,
            'noise_level': reference_noise_level,
            'undersampling_factor': reference_undersampling_factor,
        })

    deduplicated_cases = []
    seen_filepaths = set()
    for case in cases:
        filepath = build_reconstructed_filepath(
            case['source_mode'], case['lesion_intensity'], case['noise_level'], case['undersampling_factor']
        )
        if filepath in seen_filepaths:
            continue
        seen_filepaths.add(filepath)
        deduplicated_cases.append(case)
    return deduplicated_cases

generation_cases = build_generation_cases()
print('Current source tag:', build_source_tag(preview_source_mode, reference_lesion_intensity))
print('Configured reconstruction/IQ cases:', len(generation_cases))


# %%
# Step-by-step preview of the acquisition model for one image.
preview_lesion_intensity = reference_lesion_intensity if preview_source_mode == 'lesion' else None
image = build_source_image(preview_case_index, preview_source_mode, preview_lesion_intensity)
mask = sampling_masks[preview_undersampling_factor]

rng = np.random.default_rng(random_seed)
imagek = fftn(image)
imagek_shifted = fftshift(imagek)
raw_sigma = noise_level_to_raw_kspace_component_sigma(preview_noise_level)
if raw_sigma > 0:
    real_noise = rng.normal(loc=0.0, scale=raw_sigma, size=imagek.shape)
    imag_noise = rng.normal(loc=0.0, scale=raw_sigma, size=imagek.shape)
    noisek = real_noise + 1j * imag_noise
else:
    noisek = np.zeros_like(imagek)
imagek_noised_shifted = imagek_shifted + noisek
imagek_noised = ifftshift(imagek_noised_shifted)
imagek_noised_shifted_masked = imagek_noised_shifted * mask
imagek_noised_masked = ifftshift(imagek_noised_shifted_masked)
image_undersampled, _ = reconstruct_from_kspace(
    image,
    noise_level=preview_noise_level,
    undersampling_factor=preview_undersampling_factor,
    rng=np.random.default_rng(random_seed),
)

current_reconstructed_filepath = build_reconstructed_filepath(
    preview_source_mode,
    preview_lesion_intensity,
    preview_noise_level,
    preview_undersampling_factor,
)
current_iq_metrics_filepath = build_iq_metrics_filepath(
    preview_source_mode,
    preview_lesion_intensity,
    preview_noise_level,
    preview_undersampling_factor,
)
print('Preview case:', build_case_label(
    preview_source_mode, preview_lesion_intensity, preview_noise_level, preview_undersampling_factor
))
print('Preview reconstructed image file:', current_reconstructed_filepath)
print('Preview IQ metrics file:', current_iq_metrics_filepath)


# %%
plt.figure(figsize=(12, 12))
plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Current Source Image')

plt.subplot(3, 3, 2)
plt.imshow(np.log(np.abs(imagek) + 1e-10), cmap='gray')
plt.title('Log Magnitude of Original k-space')

plt.subplot(3, 3, 3)
plt.imshow(np.log(np.abs(noisek) + 1e-10), cmap='gray')
plt.title('Log Magnitude of Added k-space Noise')

plt.subplot(3, 3, 4)
plt.imshow(np.log(np.abs(imagek_noised) + 1e-10), cmap='gray')
plt.title('Log Magnitude of Noisy k-space')

plt.subplot(3, 3, 5)
plt.imshow(np.log(np.abs(imagek_noised_shifted) + 1e-10), cmap='gray')
plt.title('Shifted Noisy k-space')

plt.subplot(3, 3, 6)
plt.imshow(mask, cmap='gray')
plt.title('Sampling Mask')

plt.subplot(3, 3, 7)
plt.imshow(np.log(np.abs(imagek_noised_shifted_masked) + 1e-10), cmap='gray')
plt.title('Shifted and Masked k-space')

plt.subplot(3, 3, 8)
plt.imshow(np.log(np.abs(imagek_noised_masked) + 1e-10), cmap='gray')
plt.title('Masked k-space')

plt.subplot(3, 3, 9)
plt.imshow(image_undersampled, cmap='gray')
plt.title('Magnitude Reconstruction')

plt.tight_layout()
plt.show()


# %%
iq_results = []

for case in generation_cases:
    source_mode = case['source_mode']
    lesion_intensity = case['lesion_intensity']
    noise_level = case['noise_level']
    undersampling_factor = case['undersampling_factor']
    case_label = build_case_label(source_mode, lesion_intensity, noise_level, undersampling_factor)

    case_output_filepath = build_reconstructed_filepath(source_mode, lesion_intensity, noise_level, undersampling_factor)
    case_iq_metrics_filepath = build_iq_metrics_filepath(source_mode, lesion_intensity, noise_level, undersampling_factor)
    case_images = load_source_images(source_mode, lesion_intensity)

    rng = np.random.default_rng(random_seed)
    mse_batches = []
    ssim_batches = []
    fid = builtins.open(case_output_filepath, 'wb') if save_reconstructed_images else None
    try:
        for batch_start in range(0, n_case_images, n_batch_images):
            batch_end = min(batch_start + n_batch_images, n_case_images)
            batch_source_images = np.asarray(case_images[batch_start:batch_end], dtype=np.float32)
            batch_reconstructed_images = np.empty_like(batch_source_images)

            for local_i, source_image in enumerate(batch_source_images):
                batch_reconstructed_images[local_i], _ = reconstruct_from_kspace(
                    source_image,
                    noise_level=noise_level,
                    undersampling_factor=undersampling_factor,
                    rng=rng,
                )

            if save_iq_metrics:
                batch_mse, batch_ssim = compute_image_quality_values(batch_source_images, batch_reconstructed_images)
                mse_batches.append(batch_mse)
                ssim_batches.append(batch_ssim)
            if fid is not None:
                batch_reconstructed_images.tofile(fid)
            print(f'{case_label} reconstructed images {batch_end}/{n_case_images} done')
    finally:
        if fid is not None:
            fid.close()

    case_iq_result = {
        'source_tag': build_source_tag(source_mode, lesion_intensity),
        'noise_level': noise_level,
        'raw_kspace_component_sigma': noise_level_to_raw_kspace_component_sigma(noise_level),
        'undersampling_factor': undersampling_factor,
        'reconstructed_filepath': case_output_filepath,
    }
    if save_iq_metrics:
        iq_summary = summarize_image_quality(np.concatenate(mse_batches), np.concatenate(ssim_batches))
        np.savez(
            case_iq_metrics_filepath,
            source_tag=case_iq_result['source_tag'],
            noise_level=noise_level,
            raw_kspace_component_sigma=case_iq_result['raw_kspace_component_sigma'],
            undersampling_factor=undersampling_factor,
            **iq_summary,
        )
        case_iq_result.update(iq_summary)
        print(f'Saved IQ metrics to {case_iq_metrics_filepath}')
        print(
            f"{case_label}: "
            f"MSE={iq_summary['mse_mean']:.6f} +/- {iq_summary['mse_std']:.6f}, "
            f"SSIM={iq_summary['ssim_mean']:.4f} +/- {iq_summary['ssim_std']:.4f}"
        )
    if save_reconstructed_images:
        print(f'Saved reconstructed images to {case_output_filepath}')
    iq_results.append(case_iq_result)


# %%
# Check that the measured image-space noise matches the analytic prediction:
# with numpy's 1/N inverse FFT, each complex component has standard deviation
# noise_level * sqrt(sample_fraction / 2).
def estimate_image_space_stderr(source_mode, lesion_intensity, noise_level, undersampling_factor, image_i=0, n_trials=32, seed=0):
    image = build_source_image(image_i, source_mode, lesion_intensity)
    sampling_mask = sampling_masks[undersampling_factor]

    noiseless_magnitude, noiseless_kspace = reconstruct_from_kspace(
        image,
        noise_level=0.0,
        undersampling_factor=undersampling_factor,
        rng=np.random.default_rng(seed),
    )
    noiseless_complex = ifftn(ifftshift(noiseless_kspace))

    rng = np.random.default_rng(seed)
    complex_errors = np.empty((n_trials, *image.shape), dtype=np.complex64)
    magnitude_errors = np.empty((n_trials, *image.shape), dtype=np.float32)

    for trial_i in range(n_trials):
        noisy_magnitude, noisy_kspace = reconstruct_from_kspace(
            image,
            noise_level=noise_level,
            undersampling_factor=undersampling_factor,
            rng=rng,
        )
        complex_errors[trial_i] = ifftn(ifftshift(noisy_kspace)) - noiseless_complex
        magnitude_errors[trial_i] = noisy_magnitude - noiseless_magnitude

    sample_fraction = float(np.mean(sampling_mask))
    expected_component_stderr = noise_level * np.sqrt(sample_fraction / 2.0)

    return {
        'source_tag': build_source_tag(source_mode, lesion_intensity),
        'noise_level': noise_level,
        'raw_kspace_component_sigma': noise_level_to_raw_kspace_component_sigma(noise_level),
        'undersampling_factor': undersampling_factor,
        'sample_fraction': sample_fraction,
        'expected_component_stderr': expected_component_stderr,
        'measured_complex_real_stderr_mean': float(complex_errors.real.std(axis=0).mean()),
        'measured_complex_imag_stderr_mean': float(complex_errors.imag.std(axis=0).mean()),
        'measured_magnitude_stderr_mean': float(magnitude_errors.std(axis=0).mean()),
    }

verification_cases = [
    {'source_mode': 'lesion', 'lesion_intensity': 0.10, 'noise_level': 0.00, 'undersampling_factor': 2},
    {'source_mode': 'lesion', 'lesion_intensity': 0.10, 'noise_level': 0.00, 'undersampling_factor': 4},
    {'source_mode': 'lesion', 'lesion_intensity': 0.10, 'noise_level': 0.02, 'undersampling_factor': 2},
    {'source_mode': 'lesion', 'lesion_intensity': 0.10, 'noise_level': 0.10, 'undersampling_factor': 2},
    {'source_mode': 'lesion', 'lesion_intensity': 0.10, 'noise_level': 0.30, 'undersampling_factor': 2},
    {'source_mode': 'lesion', 'lesion_intensity': 0.20, 'noise_level': 0.10, 'undersampling_factor': 2},
]

verification_results = []
for case in verification_cases:
    verification_results.append(
        estimate_image_space_stderr(
            case['source_mode'],
            case['lesion_intensity'],
            case['noise_level'],
            case['undersampling_factor'],
            image_i=0,
            n_trials=32,
            seed=random_seed,
        )
    )

for result in verification_results:
    print(
        f"source={result['source_tag']}, normalized_noise={result['noise_level']:.2f}, "
        f"raw_kspace_sigma={result['raw_kspace_component_sigma']:.2f}, "
        f"undersample={result['undersampling_factor']}, sample_fraction={result['sample_fraction']:.3f}, "
        f"expected_component_stderr={result['expected_component_stderr']:.4f}, "
        f"real_stderr={result['measured_complex_real_stderr_mean']:.4f}, "
        f"imag_stderr={result['measured_complex_imag_stderr_mean']:.4f}, "
        f"magnitude_stderr={result['measured_magnitude_stderr_mean']:.4f}"
    )
