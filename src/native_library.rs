//! ggml keeps global callbacks and a backend registry; their DLLs must outlive models.
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use anyhow::{Context, Result, ensure};
use libloading::Library;

struct PinnedLibrary {
    library: Arc<Library>,
    _directories: Vec<DllDirectory>,
}

struct DllDirectory(usize);
#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetDefaultDllDirectories(flags: u32) -> i32;
    fn AddDllDirectory(path: *const u16) -> *mut std::ffi::c_void;
    fn RemoveDllDirectory(cookie: *mut std::ffi::c_void) -> i32;
}

impl Drop for DllDirectory {
    fn drop(&mut self) {
        unsafe {
            RemoveDllDirectory(self.0 as *mut std::ffi::c_void);
        }
    }
}

pub fn load(path: &Path) -> Result<Arc<Library>> {
    static LIBRARIES: OnceLock<Mutex<HashMap<PathBuf, PinnedLibrary>>> = OnceLock::new();
    let path = path.canonicalize()?;
    let mut libraries = LIBRARIES
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| anyhow::anyhow!("原生 DLL 状态异常，请重启服务"))?;
    if let Some(pinned) = libraries.get(&path) {
        return Ok(pinned.library.clone());
    }
    use std::os::windows::ffi::OsStrExt;
    let mut directories = Vec::new();
    unsafe {
        if SetDefaultDllDirectories(0x00001000) == 0 {
            return Err(std::io::Error::last_os_error()).context("无法设置 DLL 搜索方式");
        }
        for directory in search_directories(&path) {
            let wide: Vec<u16> = directory.as_os_str().encode_wide().chain(Some(0)).collect();
            let cookie = AddDllDirectory(wide.as_ptr());
            if cookie.is_null() {
                return Err(std::io::Error::last_os_error()).context("无法注册原生 DLL 目录");
            }
            directories.push(DllDirectory(cookie as usize));
        }
    }
    let library: Library = unsafe {
        libloading::os::windows::Library::load_with_flags(&path, 0x00000100 | 0x00001000)
    }
    .with_context(|| format!("无法加载原生 DLL 及其依赖：{}", path.display()))?
    .into();
    let library = Arc::new(library);
    libraries.insert(
        path,
        PinnedLibrary {
            library: library.clone(),
            _directories: directories,
        },
    );
    Ok(library)
}

fn search_directories(library_path: &Path) -> Vec<PathBuf> {
    let directory = library_path
        .parent()
        .expect("canonical DLL path has a parent");
    let mut directories = vec![directory.to_owned()];
    if let Some(shared) = directory
        .parent()
        .filter(|p| p.join("cudart64_12.dll").is_file())
    {
        directories.push(shared.to_owned());
    }
    directories
}

pub fn configure_cuda(backend_library: &Path, gpu: u32) -> Result<()> {
    let backend_library = backend_library.canonicalize()?;
    let directory = backend_library.parent().context("原生动态库目录无效")?;
    let path = [Some(directory), directory.parent()]
        .into_iter()
        .flatten()
        .map(|path| path.join("cudart64_12.dll"))
        .find(|path| path.is_file())
        .context("找不到配套 CUDA 运行库，请重新运行 scripts/setup.ps1")?;
    let library = load(&path)?;
    unsafe {
        let set_device = library.get::<unsafe extern "C" fn(i32) -> i32>(b"cudaSetDevice\0")?;
        let get_flags =
            library.get::<unsafe extern "C" fn(*mut u32) -> i32>(b"cudaGetDeviceFlags\0")?;
        let set_flags = library.get::<unsafe extern "C" fn(u32) -> i32>(b"cudaSetDeviceFlags\0")?;
        let status = set_device(gpu.try_into().context("CUDA 设备编号过大")?);
        ensure!(status == 0, "CUDA 设备初始化失败：{status}");
        let mut flags = 0;
        let status = get_flags(&mut flags);
        ensure!(status == 0, "读取 CUDA 等待策略失败：{status}");
        if flags & 7 == 1 {
            return Ok(());
        }
        // Keep mapping flags; use active waiting for latency-sensitive serialized inference.
        let status = set_flags((flags & !7) | 1);
        ensure!(status == 0, "设置 CUDA 等待策略失败：{status}");
        let status = get_flags(&mut flags);
        ensure!(
            status == 0 && flags & 7 == 1,
            "CUDA 等待策略未生效：{status}"
        );
        tracing::info!(gpu, flags, "CUDA 同步使用 spin 等待策略");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_cuda_directory_is_added_only_when_present() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let runtime = temporary.path().join("runtime");
        let asr = runtime.join("asr");
        std::fs::create_dir_all(&asr)?;
        let library = asr.join("transcribe.dll");
        assert_eq!(search_directories(&library), std::slice::from_ref(&asr));
        std::fs::write(runtime.join("cudart64_12.dll"), [])?;
        assert_eq!(search_directories(&library), [asr, runtime]);
        Ok(())
    }
}
