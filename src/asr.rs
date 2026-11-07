use std::{
    ffi::{CStr, CString, c_char, c_void},
    mem::{self, MaybeUninit},
    path::Path,
    ptr,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use libloading::Library;

type Handle = *mut c_void;

#[repr(C)]
struct LoadParams {
    struct_size: u64,
    backend: i32,
    device: Handle,
    no_mmap: bool,
}
#[repr(C)]
struct BackendParams {
    struct_size: u64,
    artifact_dir: *const c_char,
    allowed_backends: u32,
}
#[repr(C)]
struct SessionParams {
    struct_size: u64,
    n_threads: i32,
    kv_type: i32,
    n_ctx: i32,
}
#[repr(C)]
struct DeviceInfo {
    struct_size: u64,
    name: *const c_char,
    description: *const c_char,
    kind: *const c_char,
    device_id: *const c_char,
    memory_total: u64,
    memory_free: u64,
    device_type: i32,
}
#[repr(C)]
struct RunParams {
    struct_size: u64,
    task: i32,
    timestamps: i32,
    pnc: i32,
    itn: i32,
    diarize: i32,
    language: *const c_char,
    target_language: *const c_char,
    keep_special_tags: bool,
    family: *const c_void,
    spec_k_drafts: i32,
    vocabulary: *const *const c_char,
    n_vocabulary: i32,
    prompt: *const c_char,
    prefix: *const c_char,
}

struct Api {
    _library: Library,
    status: unsafe extern "C" fn(i32) -> *const c_char,
    load_init: unsafe extern "C" fn(*mut LoadParams),
    session_init_params: unsafe extern "C" fn(*mut SessionParams),
    run_init: unsafe extern "C" fn(*mut RunParams),
    load: unsafe extern "C" fn(*const c_char, *const LoadParams, *mut Handle) -> i32,
    session_init: unsafe extern "C" fn(Handle, *const SessionParams, *mut Handle) -> i32,
    model_free: unsafe extern "C" fn(Handle),
    session_free: unsafe extern "C" fn(Handle),
    run: unsafe extern "C" fn(Handle, *const f32, i32, *const RunParams) -> i32,
    text: unsafe extern "C" fn(Handle) -> *const c_char,
    backend: unsafe extern "C" fn(Handle) -> *const c_char,
    set_abort: unsafe extern "C" fn(Handle, Option<unsafe extern "C" fn(Handle) -> bool>, Handle),
}

pub struct Asr {
    api: Api,
    #[cfg(windows)]
    _dll_directories: Vec<DllDirectory>,
    model: Handle,
    session: Handle,
}

// The C API explicitly permits moving a session between threads; Runtime serializes access.
unsafe impl Send for Asr {}

impl Asr {
    pub fn load(library_path: &Path, model_path: &Path, gpu: usize) -> Result<Self> {
        // Only the pinned 0.4.1 ABI is accepted; never call init on an unchecked layout.
        let library_path = library_path.canonicalize().with_context(|| {
            format!(
                "找不到 transcribe 原生库：{}，请先运行安装脚本",
                library_path.display()
            )
        })?;
        #[cfg(windows)]
        let dll_directories = dll_search_directories(&library_path)
            .into_iter()
            .map(|path| DllDirectory::add(&path))
            .collect::<Result<Vec<_>>>()?;
        #[cfg(windows)]
        let library = unsafe {
            libloading::os::windows::Library::load_with_flags(
                &library_path,
                0x00000100 | 0x00001000,
            )
            .map(Library::from)
        }
        .context("无法加载 transcribe 原生库，请先运行安装脚本")?;
        #[cfg(not(windows))]
        let library = unsafe { Library::new(&library_path) }
            .context("无法加载 transcribe 原生库，请先运行安装脚本")?;
        unsafe {
            let version =
                *library.get::<unsafe extern "C" fn() -> *const c_char>(b"transcribe_version\0")?;
            ensure!(
                CStr::from_ptr(version()).to_bytes() == b"0.4.1",
                "transcribe 原生库版本必须为 0.4.1"
            );
            let abi_size = *library
                .get::<unsafe extern "C" fn(i32) -> usize>(b"transcribe_abi_struct_size\0")?;
            let abi_align = *library
                .get::<unsafe extern "C" fn(i32) -> usize>(b"transcribe_abi_struct_align\0")?;
            for (id, size, alignment) in [
                (
                    0,
                    mem::size_of::<LoadParams>(),
                    mem::align_of::<LoadParams>(),
                ),
                (
                    1,
                    mem::size_of::<SessionParams>(),
                    mem::align_of::<SessionParams>(),
                ),
                (2, mem::size_of::<RunParams>(), mem::align_of::<RunParams>()),
                (
                    13,
                    mem::size_of::<DeviceInfo>(),
                    mem::align_of::<DeviceInfo>(),
                ),
                (
                    15,
                    mem::size_of::<BackendParams>(),
                    mem::align_of::<BackendParams>(),
                ),
            ] {
                ensure!(
                    abi_size(id) == size && abi_align(id) == alignment,
                    "transcribe 原生库 ABI 结构布局不匹配：{id}"
                );
            }
            let init_params = *library.get::<unsafe extern "C" fn(*mut BackendParams)>(
                b"transcribe_backend_init_params_init\0",
            )?;
            let init_backends = *library.get::<unsafe extern "C" fn(*const BackendParams) -> i32>(
                b"transcribe_init_backends_ex\0",
            )?;
            let directory = c_path(library_path.parent().context("原生库目录无效")?)?;
            let mut backend_params = MaybeUninit::<BackendParams>::zeroed();
            init_params(backend_params.as_mut_ptr());
            let mut backend_params = backend_params.assume_init();
            backend_params.artifact_dir = directory.as_ptr();
            backend_params.allowed_backends = 0x1 | 0x8; // CPU helpers and CUDA only.
            ensure!(
                init_backends(&backend_params) == 0,
                "transcribe CUDA 后端初始化失败"
            );
            let device_count =
                *library.get::<unsafe extern "C" fn() -> i32>(b"transcribe_device_count\0")?;
            let device_get =
                *library.get::<unsafe extern "C" fn(i32) -> Handle>(b"transcribe_device_get\0")?;
            let device_info_init = *library
                .get::<unsafe extern "C" fn(*mut DeviceInfo)>(b"transcribe_device_info_init\0")?;
            let device_info = *library
                .get::<unsafe extern "C" fn(Handle, *mut DeviceInfo) -> i32>(
                    b"transcribe_device_get_info\0",
                )?;
            let mut devices = Vec::new();
            for index in 0..device_count() {
                let handle = device_get(index);
                if handle.is_null() {
                    continue;
                }
                let mut info = MaybeUninit::<DeviceInfo>::zeroed();
                device_info_init(info.as_mut_ptr());
                if device_info(handle, info.as_mut_ptr()) == 0 {
                    let info = info.assume_init();
                    if !info.kind.is_null() && CStr::from_ptr(info.kind).to_bytes() == b"cuda" {
                        devices.push(handle);
                    }
                }
            }
            let device = *devices
                .get(gpu)
                .context("ASR 未发现指定的 CUDA GPU，请检查驱动和 --gpu")?;
            let api = Api {
                status: *library.get(b"transcribe_status_string\0")?,
                load_init: *library.get(b"transcribe_model_load_params_init\0")?,
                session_init_params: *library.get(b"transcribe_session_params_init\0")?,
                run_init: *library.get(b"transcribe_run_params_init\0")?,
                load: *library.get(b"transcribe_model_load_file\0")?,
                session_init: *library.get(b"transcribe_session_init\0")?,
                model_free: *library.get(b"transcribe_model_free\0")?,
                session_free: *library.get(b"transcribe_session_free\0")?,
                run: *library.get(b"transcribe_run\0")?,
                text: *library.get(b"transcribe_full_text\0")?,
                backend: *library.get(b"transcribe_model_backend\0")?,
                set_abort: *library.get(b"transcribe_set_abort_callback\0")?,
                _library: library,
            };
            let mut params = MaybeUninit::<LoadParams>::zeroed();
            (api.load_init)(params.as_mut_ptr());
            let mut params = params.assume_init();
            params.backend = 5; // TRANSCRIBE_BACKEND_CUDA; no AUTO/CPU fallback.
            params.device = device;
            let mut asr = Self {
                api,
                #[cfg(windows)]
                _dll_directories: dll_directories,
                model: ptr::null_mut(),
                session: ptr::null_mut(),
            };
            let file = c_path(model_path)?;
            let status = (asr.api.load)(file.as_ptr(), &params, &mut asr.model);
            asr.check(status, "ASR 模型加载")?;
            ensure!(!asr.model.is_null(), "ASR 原生库未返回模型句柄");
            let backend = (asr.api.backend)(asr.model);
            ensure!(!backend.is_null(), "ASR 原生库未返回计算后端");
            let backend = CStr::from_ptr(backend).to_string_lossy();
            ensure!(
                backend.to_ascii_lowercase().contains("cuda"),
                "ASR 未运行在 CUDA：{backend}"
            );
            let mut session = MaybeUninit::<SessionParams>::zeroed();
            (asr.api.session_init_params)(session.as_mut_ptr());
            let mut session = session.assume_init();
            session.n_ctx = 2048;
            let status = (asr.api.session_init)(asr.model, &session, &mut asr.session);
            asr.check(status, "ASR 会话创建")?;
            ensure!(!asr.session.is_null(), "ASR 原生库未返回会话句柄");
            tracing::info!(%backend,"Fun-ASR-Nano BF16 原生推理已就绪");
            Ok(asr)
        }
    }

    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: &str,
        timeout: Duration,
    ) -> Result<String> {
        ensure!(
            !samples.is_empty() && samples.len() <= i32::MAX as usize,
            "音频采样数为空或超出原生接口限制"
        );
        let language = CString::new(language)?;
        unsafe {
            let mut params = MaybeUninit::<RunParams>::zeroed();
            (self.api.run_init)(params.as_mut_ptr());
            let mut params = params.assume_init();
            params.timestamps = 0;
            params.itn = 2; // TRANSCRIBE_ITN_MODE_ON.
            if language.to_bytes() != b"auto" {
                params.language = language.as_ptr();
            }
            let deadline = Instant::now()
                .checked_add(timeout)
                .context("ASR 超时时间过大")?;
            (self.api.set_abort)(
                self.session,
                Some(abort_at_deadline),
                (&deadline as *const Instant).cast_mut().cast(),
            );
            let status = (self.api.run)(
                self.session,
                samples.as_ptr(),
                samples.len() as i32,
                &params,
            );
            (self.api.set_abort)(self.session, None, ptr::null_mut());
            self.check(status, "ASR 推理")?;
            let text = (self.api.text)(self.session);
            ensure!(!text.is_null(), "ASR 未返回文本");
            Ok(CStr::from_ptr(text)
                .to_str()
                .context("ASR 返回了无效 UTF-8")?
                .trim()
                .to_owned())
        }
    }

    fn check(&self, status: i32, action: &str) -> Result<()> {
        if status != 0 {
            let message = unsafe { CStr::from_ptr((self.api.status)(status)) }.to_string_lossy();
            anyhow::bail!("{action}失败：{message}（{status}）");
        }
        Ok(())
    }
}

unsafe extern "C" fn abort_at_deadline(data: Handle) -> bool {
    Instant::now() >= unsafe { *(data.cast::<Instant>()) }
}

impl Drop for Asr {
    fn drop(&mut self) {
        unsafe {
            (self.api.session_free)(self.session);
            (self.api.model_free)(self.model);
        }
    }
}

fn c_path(path: &Path) -> Result<CString> {
    CString::new(path.to_str().context("路径必须为有效 UTF-8")?).context("路径包含 NUL 字符")
}

#[cfg(any(windows, test))]
fn dll_search_directories(library: &Path) -> Vec<std::path::PathBuf> {
    let directory = library
        .parent()
        .expect("canonical library path has a parent");
    let mut directories = vec![directory.to_owned()];
    if let Some(shared) = directory
        .parent()
        .filter(|path| path.join("cudart64_12.dll").is_file())
    {
        directories.push(shared.to_owned());
    }
    directories
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_cuda_search_keeps_backend_libraries_isolated() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let runtime = temporary.path().join("runtime");
        let asr = runtime.join("asr");
        std::fs::create_dir_all(&asr)?;
        let library = asr.join("transcribe.dll");
        assert_eq!(dll_search_directories(&library), std::slice::from_ref(&asr));
        std::fs::write(runtime.join("cudart64_12.dll"), [])?;
        assert_eq!(dll_search_directories(&library), [asr, runtime]);
        Ok(())
    }
}

#[cfg(windows)]
struct DllDirectory(Handle);

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetDefaultDllDirectories(flags: u32) -> i32;
    fn AddDllDirectory(path: *const u16) -> Handle;
    fn RemoveDllDirectory(cookie: Handle) -> i32;
}

#[cfg(windows)]
impl DllDirectory {
    fn add(path: &Path) -> Result<Self> {
        use std::os::windows::ffi::OsStrExt;
        let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        // ggml loads its CUDA plugin itself; its dependent DLLs need the registered directory.
        unsafe {
            if SetDefaultDllDirectories(0x00001000) == 0 {
                return Err(std::io::Error::last_os_error()).context("无法设置 ASR DLL 搜索方式");
            }
            let cookie = AddDllDirectory(path.as_ptr());
            if cookie.is_null() {
                return Err(std::io::Error::last_os_error()).context("无法注册 ASR DLL 目录");
            }
            Ok(Self(cookie))
        }
    }
}

#[cfg(windows)]
impl Drop for DllDirectory {
    fn drop(&mut self) {
        unsafe {
            RemoveDllDirectory(self.0);
        }
    }
}
