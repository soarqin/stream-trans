//! Typed entry in our pinned llama-server-impl.dll; no CLI main or process globals.
use std::{
    ffi::{CStr, CString, c_char, c_void},
    path::Path,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

use anyhow::{Context, Result, ensure};
use libloading::Library;

use super::StartupLogs;

#[repr(C)]
struct Config {
    abi_version: u32,
    struct_size: u32,
    model_path: *const c_char,
    api_key: *const c_char,
    gpu: u32,
    port: u32,
    context_size: u32,
}

#[repr(C)]
struct Receipt {
    abi_version: u32,
    struct_size: u32,
    state: u32,
    result: i32,
    build_number: i32,
    gpu_layers: u32,
    total_layers: u32,
    device: [c_char; 64],
    message: [c_char; 512],
}

struct Api {
    _library: Arc<Library>,
    run: unsafe extern "C" fn(*mut c_void) -> i32,
    stop: unsafe extern "C" fn(*mut c_void),
    status: unsafe extern "C" fn(*mut c_void, *mut Receipt) -> i32,
    destroy: unsafe extern "C" fn(*mut c_void),
}

struct Instance {
    api: Api,
    handle: *mut c_void,
}

// Only run's dedicated thread mutates the model; status/stop are synchronized by the ABI.
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { (self.api.destroy)(self.handle) };
    }
}

pub(super) struct ServerProcess {
    instance: Arc<Instance>,
    worker: Option<JoinHandle<i32>>,
    pub(super) logs: Arc<Mutex<StartupLogs>>,
}

impl ServerProcess {
    pub(super) fn spawn(
        path: &Path,
        model: &Path,
        gpu: usize,
        port: u16,
        context_size: u32,
        api_key: String,
    ) -> Result<Self> {
        let library = crate::native_library::load(path)?;
        let model_path = CString::new(model.to_str().context("模型路径不是有效 UTF-8")?)?;
        let key = CString::new(api_key.as_str())?;
        let config = Config {
            abi_version: 1,
            struct_size: size_of::<Config>() as u32,
            model_path: model_path.as_ptr(),
            api_key: key.as_ptr(),
            gpu: gpu.try_into().context("CUDA 设备编号过大")?,
            port: u32::from(port),
            context_size,
        };
        let instance = unsafe {
            let abi = library
                .get::<unsafe extern "C" fn() -> u32>(b"streamtrans_server_abi_version\0")
                .context(
                    "llama-server-impl.dll 没有进程内入口，请运行 scripts/setup-llama.ps1 重编",
                )?;
            ensure!(abi() == 1, "llama-server-impl.dll 进程内 ABI 不匹配");
            let create =
                *library
                    .get::<unsafe extern "C" fn(*const Config, *mut c_char, usize) -> *mut c_void>(
                        b"streamtrans_server_create\0",
                    )?;
            let api = Api {
                run: *library.get(b"streamtrans_server_run\0")?,
                stop: *library.get(b"streamtrans_server_request_stop\0")?,
                status: *library.get(b"streamtrans_server_status\0")?,
                destroy: *library.get(b"streamtrans_server_destroy\0")?,
                _library: library,
            };
            let mut error = [0; 1024];
            let handle = create(&config, error.as_mut_ptr(), error.len());
            error[error.len() - 1] = 0;
            ensure!(
                !handle.is_null(),
                "无法创建进程内翻译服务：{}",
                CStr::from_ptr(error.as_ptr()).to_string_lossy()
            );
            Arc::new(Instance { api, handle })
        };
        let runner = instance.clone();
        let worker = thread::Builder::new()
            .name("llama-server-embedded".to_owned())
            .spawn(move || unsafe { (runner.api.run)(runner.handle) })
            .context("无法启动进程内翻译服务线程")?;
        Ok(Self {
            instance,
            worker: Some(worker),
            logs: Arc::new(Mutex::new(StartupLogs {
                gpu,
                api_key,
                ..Default::default()
            })),
        })
    }

    pub(super) fn running(&mut self) -> bool {
        if self.worker.is_none() {
            return false;
        }
        let mut receipt: Receipt = unsafe { std::mem::zeroed() };
        receipt.abi_version = 1;
        receipt.struct_size = size_of::<Receipt>() as u32;
        let status = unsafe { (self.instance.api.status)(self.instance.handle, &mut receipt) };
        receipt.device[63] = 0;
        receipt.message[511] = 0;
        if let Ok(mut logs) = self.logs.lock() {
            logs.build_confirmed = receipt.build_number == 11960;
            let device = unsafe { CStr::from_ptr(receipt.device.as_ptr()) }.to_string_lossy();
            logs.cuda_confirmed = device == format!("CUDA{}", logs.gpu);
            logs.offloaded = Some((receipt.gpu_layers as usize, receipt.total_layers as usize));
            let message = unsafe { CStr::from_ptr(receipt.message.as_ptr()) }.to_string_lossy();
            if status != 0 || receipt.state == 4 {
                logs.record(&format!("error: embedded status {status}: {message}"));
            }
        }
        status == 0 && receipt.state <= 2 && !self.worker.as_ref().is_some_and(|w| w.is_finished())
    }

    pub(super) fn stop(&mut self) -> Result<()> {
        if let Some(worker) = self.worker.take() {
            unsafe { (self.instance.api.stop)(self.instance.handle) };
            // CUDA has no hard kernel cancellation. Retain the GPU lock until native work exits.
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("进程内翻译服务线程异常退出"))?;
        }
        Ok(())
    }

    pub(super) fn diagnostics(&self) -> String {
        self.logs
            .lock()
            .map(|logs| logs.diagnostics())
            .unwrap_or_else(|_| "启动诊断状态异常".to_owned())
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_c_abi_layout() {
        assert_eq!(size_of::<Config>(), 40);
        assert_eq!(std::mem::align_of::<Config>(), 8);
        assert_eq!(size_of::<Receipt>(), 604);
        assert_eq!(std::mem::align_of::<Receipt>(), 4);
    }

    #[test]
    #[ignore = "requires the built embedded llama runtime; does not load model weights"]
    fn native_entry_validates_abi_and_stop_before_run_survives_restart() -> Result<()> {
        let library =
            crate::native_library::load(Path::new(".tools/llama-embedded/llama-server-impl.dll"))?;
        unsafe {
            let create =
                *library
                    .get::<unsafe extern "C" fn(*const Config, *mut c_char, usize) -> *mut c_void>(
                        b"streamtrans_server_create\0",
                    )?;
            let run = *library
                .get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"streamtrans_server_run\0")?;
            let stop = *library
                .get::<unsafe extern "C" fn(*mut c_void)>(b"streamtrans_server_request_stop\0")?;
            let status = *library.get::<unsafe extern "C" fn(*mut c_void, *mut Receipt) -> i32>(
                b"streamtrans_server_status\0",
            )?;
            let destroy = *library
                .get::<unsafe extern "C" fn(*mut c_void)>(b"streamtrans_server_destroy\0")?;
            let model = CString::new("missing-model.gguf")?;
            let key = CString::new("native-test-key")?;
            let mut config = Config {
                abi_version: 0,
                struct_size: size_of::<Config>() as u32,
                model_path: model.as_ptr(),
                api_key: key.as_ptr(),
                gpu: 0,
                port: 39999,
                context_size: 4096,
            };
            let mut error = [0; 1024];
            assert!(create(&config, error.as_mut_ptr(), error.len()).is_null());
            config.abi_version = 1;
            for _ in 0..3 {
                let handle = create(&config, error.as_mut_ptr(), error.len());
                assert!(!handle.is_null());
                stop(handle);
                stop(handle);
                assert_eq!(run(handle), 0);
                assert_eq!(run(handle), -1);
                let mut receipt: Receipt = std::mem::zeroed();
                assert_eq!(status(handle, &mut receipt), -1);
                receipt.abi_version = 1;
                receipt.struct_size = size_of::<Receipt>() as u32;
                assert_eq!(status(handle, &mut receipt), 0);
                assert_eq!(receipt.state, 3);
                assert_eq!(receipt.result, 0);
                destroy(handle);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires built runtime, CUDA and STREAMTRANS_TEST_LLM_MODEL"]
    fn native_model_load_cancellation_and_model_restart() -> Result<()> {
        let model = std::env::var_os("STREAMTRANS_TEST_LLM_MODEL")
            .context("请设置 STREAMTRANS_TEST_LLM_MODEL 为 Hy-MT2 Q6_K 文件路径")?;
        let library = Path::new(".tools/llama-embedded/llama-server-impl.dll");
        for _ in 0..2 {
            let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
            let port = listener.local_addr()?.port();
            drop(listener);
            let mut server = ServerProcess::spawn(
                library,
                Path::new(&model),
                0,
                port,
                4096,
                "cancel-test".to_owned(),
            )?;
            thread::sleep(std::time::Duration::from_millis(150));
            server.stop()?;
            assert!(!server.running());
        }
        // Reload in the same process after cancelled loads and fully released models.
        for _ in 0..2 {
            let mut server = super::super::Llm::load(
                library,
                Path::new(&model),
                0,
                4096,
                2048,
                std::time::Duration::from_secs(30),
                &std::sync::atomic::AtomicBool::new(false),
            )?;
            assert!(!server.translate("今天下午三点开会。", "en")?.is_empty());
        }
        Ok(())
    }
}
