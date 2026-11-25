//! Run Core ML models from Rust: load a model and predict on float32 tensors, one input and one output.
//!
//! Every call into Core ML is in this file, behind a safe API. Models are loaded from a compiled `.mlmodelc`, or
//! from an `.mlpackage` / `.mlmodel`, which is compiled first (into a temporary directory Core ML chooses).

use anyhow::{Context, Result, bail, ensure};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel, MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use std::path::Path;

/// Which processors Core ML may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compute {
    All,
    CpuAndGpu,
    CpuOnly,
    CpuAndNeuralEngine,
}

/// A loaded model.
pub struct Model {
    model: Retained<MLModel>,
}

// SAFETY: MLModel is documented as safe to use from multiple threads for predictions; `predict` takes `&self`
// and builds its inputs and outputs per call.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

fn err(e: Retained<NSError>) -> anyhow::Error {
    anyhow::anyhow!("{}", e.localizedDescription())
}

impl Model {
    pub fn load(path: &Path, compute: Compute) -> Result<Self> {
        ensure!(path.exists(), "no model at {}", path.display());
        let s = NSString::from_str(&path.to_string_lossy());
        let url = NSURL::fileURLWithPath(&s);
        let compiled = if path.extension().is_some_and(|e| e == "mlmodelc") {
            url
        } else {
            // SAFETY: a file URL; the result is a new URL or an error.
            #[allow(deprecated)]
            unsafe { MLModel::compileModelAtURL_error(&url) }.map_err(err).with_context(|| format!("compiling {}", path.display()))?
        };
        // SAFETY: plain object construction and a setter with a valid enum value.
        let config = unsafe { MLModelConfiguration::new() };
        let units = match compute {
            Compute::All => MLComputeUnits::All,
            Compute::CpuAndGpu => MLComputeUnits::CPUAndGPU,
            Compute::CpuOnly => MLComputeUnits::CPUOnly,
            Compute::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
        };
        unsafe { config.setComputeUnits(units) };
        // SAFETY: a compiled-model URL and a configuration, both live for the call.
        let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&compiled, &config) }
            .map_err(err)
            .with_context(|| format!("loading {}", path.display()))?;
        Ok(Self { model })
    }

    /// Runs the model on one float32 tensor of `shape` (row-major `data`) as input `input`, and returns output
    /// `output` as float32 values (row-major).
    pub fn predict(&self, input: &str, shape: &[usize], data: &[f32], output: &str) -> Result<Vec<f32>> {
        ensure!(shape.iter().product::<usize>() == data.len(), "{} values for shape {shape:?}", data.len());
        let dims: Vec<Retained<NSNumber>> = shape.iter().map(|&d| NSNumber::new_usize(d)).collect();
        let shape_ns = NSArray::from_retained_slice(&dims);
        // SAFETY: allocation with a valid shape and data type; the array owns contiguous row-major storage.
        let array = unsafe { MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &shape_ns, MLMultiArrayDataType::Float32) }.map_err(err)?;
        // SAFETY: the array was just created Float32 with exactly `data.len()` elements, contiguous; nothing else
        // holds it yet.
        #[allow(deprecated)]
        unsafe {
            let dst = array.dataPointer().as_ptr().cast::<f32>();
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        // SAFETY: wrapping a live multi-array.
        let value = unsafe { MLFeatureValue::featureValueWithMultiArray(&array) };
        let key = NSString::from_str(input);
        let dict: Retained<NSDictionary<NSString, AnyObject>> =
            NSDictionary::from_retained_objects(&[&*key], &[Retained::into_super(Retained::into_super(value))]);
        // SAFETY: a dictionary of NSString to MLFeatureValue, as the initializer requires.
        let provider = unsafe { MLDictionaryFeatureProvider::initWithDictionary_error(MLDictionaryFeatureProvider::alloc(), &dict) }.map_err(err)?;
        let provider = ProtocolObject::<dyn MLFeatureProvider>::from_ref(&*provider);
        // SAFETY: a loaded model and a live feature provider; synchronous.
        let out = unsafe { self.model.predictionFromFeatures_error(provider) }.map_err(err).context("Core ML prediction")?;
        // SAFETY: reading a named feature from the result.
        let value = unsafe { out.featureValueForName(&NSString::from_str(output)) }.with_context(|| format!("no output {output:?}"))?;
        let array = unsafe { value.multiArrayValue() }.with_context(|| format!("output {output:?} is not a tensor"))?;
        read_f32(&array)
    }
}

fn read_f32(array: &MLMultiArray) -> Result<Vec<f32>> {
    // SAFETY: reading the type, count and contiguous storage of a live array we hold.
    unsafe {
        let n = usize::try_from(array.count())?;
        #[allow(deprecated)]
        let ptr = array.dataPointer().as_ptr();
        match array.dataType() {
            MLMultiArrayDataType::Float32 => Ok(std::slice::from_raw_parts(ptr.cast::<f32>(), n).to_vec()),
            MLMultiArrayDataType::Float16 => Ok(std::slice::from_raw_parts(ptr.cast::<u16>(), n).iter().map(|&h| f16_to_f32(h)).collect()),
            MLMultiArrayDataType::Double => Ok(std::slice::from_raw_parts(ptr.cast::<f64>(), n).iter().map(|&d| d as f32).collect()),
            t => bail!("unsupported output type {t:?}"),
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, frac) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = match exp {
        0 if frac == 0 => sign << 31,
        0 => {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => (sign << 31) | 0x7f80_0000 | (frac << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333).abs() < 1e-3);
    }

    /// Set `COREML_TEST_MODEL` to an .mlpackage with input "x" [1,4] and output "y" (e.g. from
    /// `tests/make_test_model.py`) to run.
    #[test]
    fn predicts_with_a_model() {
        let Some(p) = std::env::var_os("COREML_TEST_MODEL") else { return };
        let m = Model::load(Path::new(&p), Compute::CpuOnly).unwrap();
        let y = m.predict("x", &[1, 4], &[1.0, 2.0, 3.0, 4.0], "y").unwrap();
        assert_eq!(y.len(), 4);
        assert!((y[0] - 2.0).abs() < 1e-3 && (y[3] - 8.0).abs() < 1e-3, "{y:?}");
    }
}
