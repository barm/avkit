//! Thread priority (Darwin).

/// Put the calling thread in the background band (lower CPU and I/O priority), for work that must never slow a
/// recording down, like the break scan's decoding or a repair.
pub fn background() {
    // SAFETY: setpriority on the calling thread (PRIO_DARWIN_THREAD, who = 0) with PRIO_DARWIN_BG; no pointers.
    unsafe {
        libc::setpriority(libc::PRIO_DARWIN_THREAD, 0, libc::PRIO_DARWIN_BG);
    }
}
