
import SimpleITK as sitk

ref = sitk.ReadImage(r"E:\msc_data\ismrm_2025\dB0_035\anat\t1w\T1w_MP2RAGE_brain_20250228140637_8.nii.gz")
mov = sitk.ReadImage(r"E:\msc_data\ismrm_2025\dB0_035\anat\t1w\T1w_MP2RAGE_cervical_20250228140637_10.nii.gz")

matcher = sitk.HistogramMatchingImageFilter()
matcher.SetNumberOfHistogramLevels(256)
matcher.SetNumberOfMatchPoints(15)
matcher.ThresholdAtMeanIntensityOn()

mov_norm = matcher.Execute(mov, ref)
sitk.WriteImage(mov_norm, r"E:\msc_data\ismrm_2025\dB0_035\anat\t1w\/normalized_cervical.nii.gz")
