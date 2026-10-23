//! Machine-wide readings for the Monitor: GPU load, memory, temperature and
//! power from the kernel's `amdgpu` sysfs files, and system RAM from
//! `/proc/meminfo`.
//!
//! These describe the machine, not a session. Two servers sharing a GPU see the
//! same figures, which is why the Monitor labels them as system-wide.
//!
//! Plain file reads, no vendor tools: `amdgpu` publishes everything needed
//! under `/sys/class/drm/card*/device/`. Other vendors report nothing here and
//! the Monitor simply leaves the GPU section out. Reading `gpu_busy_percent`
//! wakes a runtime-suspended discrete GPU, so the caller samples only while a
//! session is live.

use std::path::Path;

use super::history::Series;

/// One reading of one GPU. Every figure is optional: drivers and kernels differ
/// in what they expose, and a missing file is not a zero.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuSample {
    /// `amdgpu 1002:1586` — the driver and PCI id, which is what sysfs knows.
    pub name: String,
    pub busy_percent: Option<f64>,
    /// Dedicated memory, used and total bytes.
    pub vram: Option<(u64, u64)>,
    /// System memory the GPU has mapped, used and total bytes. On an APU this
    /// is where the model lives; on a discrete card it is usually near empty.
    pub gtt: Option<(u64, u64)>,
    pub temp_c: Option<f64>,
    pub power_w: Option<f64>,
    /// Shader clock, which drops to idle within a second of the work stopping.
    pub clock_mhz: Option<f64>,
}

/// Physical memory, from `/proc/meminfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemInfo {
    pub total: u64,
    pub available: u64,
}

impl MemInfo {
    pub fn used(&self) -> u64 {
        self.total.saturating_sub(self.available)
    }
}

/// The latest machine reading plus a minute of history for the GPU tiles.
#[derive(Debug, Default)]
pub struct HostMonitor {
    /// The GPU the tiles follow — see [`primary`] — and how many there are.
    pub gpu: Option<GpuSample>,
    pub gpu_count: usize,
    pub gpu_busy: Series,
    pub gpu_temp: Series,
    pub gpu_power: Series,
    pub gpu_clock: Series,
    pub memory: Option<MemInfo>,
}

impl HostMonitor {
    /// Take one reading from the live system.
    pub fn sample(&mut self) {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok();
        self.sample_from(Path::new("/sys/class/drm"), meminfo.as_deref());
    }

    fn sample_from(&mut self, drm: &Path, meminfo: Option<&str>) {
        let gpus = gpus(drm);
        self.gpu_count = gpus.len();
        self.gpu = primary(gpus);
        let gpu = self.gpu.as_ref();
        self.gpu_busy.push(gpu.and_then(|g| g.busy_percent));
        self.gpu_temp.push(gpu.and_then(|g| g.temp_c));
        self.gpu_power.push(gpu.and_then(|g| g.power_w));
        self.gpu_clock.push(gpu.and_then(|g| g.clock_mhz));
        self.memory = meminfo.and_then(parse_meminfo);
    }
}

/// Every GPU `amdgpu` reports under `drm`, in card order.
fn gpus(drm: &Path) -> Vec<GpuSample> {
    let Ok(entries) = std::fs::read_dir(drm) else { return Vec::new() };
    let mut cards: Vec<(u32, std::path::PathBuf)> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            // `card1` is a GPU; `card1-DP-1` is one of its connectors.
            let name = entry.file_name().into_string().ok()?;
            let index = name.strip_prefix("card")?.parse().ok()?;
            Some((index, entry.path().join("device")))
        })
        .collect();
    cards.sort_by_key(|(index, _)| *index);
    cards.iter().filter_map(|(_, device)| read_amdgpu(device)).collect()
}

/// The GPU worth following when there are several: the one with the most
/// dedicated memory, which is where an inference server is put. An APU's
/// carve-out is small, so a discrete card wins over it.
fn primary(gpus: Vec<GpuSample>) -> Option<GpuSample> {
    gpus.into_iter().max_by_key(|gpu| gpu.vram.map_or(0, |(_, total)| total))
}

/// One `amdgpu` device directory, or `None` if it is not one.
fn read_amdgpu(device: &Path) -> Option<GpuSample> {
    // The file that marks an amdgpu device; the other vendors have no such
    // file, and neither do display-only devices.
    let busy = read_number(&device.join("gpu_busy_percent"))?;
    let pair = |used: &str, total: &str| {
        Some((read_number(&device.join(used))? as u64, read_number(&device.join(total))? as u64))
    };
    let hwmon = std::fs::read_dir(device.join("hwmon"))
        .ok()
        .and_then(|mut dirs| dirs.find_map(Result::ok))
        .map(|entry| entry.path());
    let hwmon_value = |file: &str| hwmon.as_ref().and_then(|dir| read_number(&dir.join(file)));
    let pci_id = std::fs::read_to_string(device.join("uevent")).ok().and_then(|uevent| {
        uevent.lines().find_map(|line| line.strip_prefix("PCI_ID=")).map(str::to_lowercase)
    });

    Some(GpuSample {
        name: match pci_id {
            Some(id) => format!("amdgpu {id}"),
            None => "amdgpu".into(),
        },
        busy_percent: Some(busy),
        vram: pair("mem_info_vram_used", "mem_info_vram_total"),
        gtt: pair("mem_info_gtt_used", "mem_info_gtt_total"),
        // hwmon reports millidegrees, microwatts and hertz. `power1_average` is the
        // smoothed figure; newer kernels on some parts only have the instant.
        temp_c: hwmon_value("temp1_input").map(|milli| milli / 1000.0),
        clock_mhz: hwmon_value("freq1_input").map(|hz| hz / 1_000_000.0),
        power_w: hwmon_value("power1_average")
            .or_else(|| hwmon_value("power1_input"))
            .map(|micro| micro / 1_000_000.0),
    })
}

fn read_number(path: &Path) -> Option<f64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// `MemTotal` and `MemAvailable`, which `/proc/meminfo` gives in KiB.
fn parse_meminfo(text: &str) -> Option<MemInfo> {
    let field = |name: &str| -> Option<u64> {
        let line = text.lines().find(|line| line.starts_with(name))?;
        let kib: u64 = line[name.len()..].trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kib * 1024)
    };
    Some(MemInfo { total: field("MemTotal:")?, available: field("MemAvailable:")? })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("llmctl-host-{tag}-{nonce}"))
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn meminfo_gives_total_and_available_in_bytes() {
        let text = "MemTotal:       130023424 kB\nMemFree:         1000 kB\nMemAvailable:   116391936 kB\n";
        let mem = parse_meminfo(text).unwrap();
        assert_eq!(mem.total, 130_023_424 * 1024);
        assert_eq!(mem.used(), (130_023_424 - 116_391_936) * 1024);
        assert_eq!(parse_meminfo("MemTotal: 1 kB\n"), None, "no available figure, no reading");
    }

    /// Laid out as a Strix Halo reports itself: a small VRAM carve-out, the
    /// model in GTT, and a connector entry beside the card that is not a GPU.
    #[test]
    fn an_amdgpu_card_is_read_from_sysfs_and_its_connectors_are_skipped() {
        let drm = temp_root("amdgpu");
        let device = drm.join("card1/device");
        write(&device.join("gpu_busy_percent"), "94\n");
        write(&device.join("mem_info_vram_used"), "528224256\n");
        write(&device.join("mem_info_vram_total"), "536870912\n");
        write(&device.join("mem_info_gtt_used"), "43423322112\n");
        write(&device.join("mem_info_gtt_total"), "133143986176\n");
        write(&device.join("uevent"), "DRIVER=amdgpu\nPCI_ID=1002:1586\n");
        write(&device.join("hwmon/hwmon7/temp1_input"), "64000\n");
        write(&device.join("hwmon/hwmon7/power1_input"), "118077000\n");
        write(&device.join("hwmon/hwmon7/freq1_input"), "2788000000\n");
        std::fs::create_dir_all(drm.join("card1-DP-1")).unwrap();

        let mut host = HostMonitor::default();
        host.sample_from(&drm, None);
        let gpu = host.gpu.clone().expect("the card is read");
        assert_eq!(host.gpu_count, 1, "the connector is not a second GPU");
        assert_eq!(gpu.name, "amdgpu 1002:1586");
        assert_eq!(gpu.busy_percent, Some(94.0));
        assert_eq!(gpu.gtt, Some((43_423_322_112, 133_143_986_176)));
        assert_eq!(gpu.temp_c, Some(64.0));
        assert_eq!(gpu.power_w, Some(118.077), "power1_input stands in for a missing average");
        assert_eq!(gpu.clock_mhz, Some(2788.0));
        assert_eq!(host.gpu_busy.last(), Some(94.0));
        assert_eq!(host.memory, None, "no meminfo, no RAM figure");

        let _ = std::fs::remove_dir_all(drm);
    }

    #[test]
    fn the_discrete_card_is_followed_over_an_apu() {
        let drm = temp_root("primary");
        for (card, vram) in [("card0", "536870912"), ("card1", "17179869184")] {
            let device = drm.join(card).join("device");
            write(&device.join("gpu_busy_percent"), "0\n");
            write(&device.join("mem_info_vram_used"), "0\n");
            write(&device.join("mem_info_vram_total"), vram);
        }
        // A device without `gpu_busy_percent` is some other vendor's.
        write(&drm.join("card2/device/vendor"), "0x10de\n");

        let mut host = HostMonitor::default();
        host.sample_from(&drm, None);
        assert_eq!(host.gpu_count, 2);
        assert_eq!(host.gpu.unwrap().vram, Some((0, 17_179_869_184)));

        let _ = std::fs::remove_dir_all(drm);
    }

    #[test]
    fn no_gpu_leaves_a_gap_in_the_history() {
        let mut host = HostMonitor::default();
        host.sample_from(Path::new("/nonexistent/llmctl/drm"), None);
        assert!(host.gpu.is_none());
        assert_eq!(host.gpu_busy.values(), vec![None]);
    }
}
