//! Limit worker inheritance even during Darwin's recvmsg/F_SETFD window.
use std::{io, os::unix::process::CommandExt, process::Command};

pub(super) fn configure(command: &mut Command) -> io::Result<()> {
    // Resolve both syscall wrappers before fork, including on the first launch.
    // The invalid descriptor makes this fcntl call harmless to the parent's fds.
    unsafe {
        libc::fcntl(-1, libc::F_GETFD);
    }
    // Allocate before fork. The kernel estimate
    // includes spare slots; add room for Command's pipes and concurrent opens.
    let bytes = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDLISTFDS,
            0,
            std::ptr::null_mut(),
            0,
        )
    };
    if bytes <= 0 {
        return Err(io::Error::last_os_error());
    }
    let count = (bytes as usize).div_ceil(size_of::<libc::proc_fdinfo>()) + 64;
    let mut descriptors = Vec::new();
    descriptors
        .try_reserve_exact(count)
        .map_err(|_| io::Error::from_raw_os_error(libc::ENOMEM))?;
    descriptors.resize(
        count,
        libc::proc_fdinfo {
            proc_fd: 0,
            proc_fdtype: 0,
        },
    );
    // SAFETY: the callback performs only the audited Darwin syscall wrapper,
    // fcntl and operations on preallocated memory. It neither allocates nor
    // locks. Register it before the callback that installs the asset socket.
    unsafe {
        command.pre_exec(move || mark_inherited(&mut descriptors));
    }
    Ok(())
}

/// Must run only in the forked child, before exec and before installing fd 3.
unsafe fn mark_inherited(descriptors: &mut [libc::proc_fdinfo]) -> io::Result<()> {
    let capacity = i32::try_from(size_of_val(descriptors))
        .map_err(|_| io::Error::from_raw_os_error(libc::EOVERFLOW))?;
    // proc_pidinfo is a thin __proc_info syscall wrapper on Darwin:
    // https://github.com/apple-oss-distributions/xnu/blob/main/libsyscall/wrappers/libproc/libproc.c
    // Unlike a parent-side snapshot, this list includes descriptors received
    // concurrently with fork. No other application thread exists in the child.
    let bytes = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDLISTFDS,
            0,
            descriptors.as_mut_ptr().cast(),
            capacity,
        )
    };
    if bytes <= 0 {
        return Err(io::Error::last_os_error());
    }
    if bytes >= capacity || !(bytes as usize).is_multiple_of(size_of::<libc::proc_fdinfo>()) {
        // A full list might be truncated. Fail this spawn instead of allowing
        // unchecked inheritance or allocating after fork.
        return Err(io::Error::from_raw_os_error(libc::ENOBUFS));
    }
    for descriptor in &descriptors[..bytes as usize / size_of::<libc::proc_fdinfo>()] {
        if descriptor.proc_fd >= 3
            && unsafe { libc::fcntl(descriptor.proc_fd, libc::F_SETFD, libc::FD_CLOEXEC) } == -1
        {
            return Err(io::Error::last_os_error());
        }
    }
    // Mark rather than close: Rust still needs its private exec-error pipe
    // if exec fails. The subsequent dup2 explicitly permits only the asset fd.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_descriptor_list_rejects_spawn_through_the_exec_error_pipe() {
        let mut descriptors = [libc::proc_fdinfo {
            proc_fd: 0,
            proc_fdtype: 0,
        }];
        let mut command = Command::new("/usr/bin/true");
        unsafe {
            command.pre_exec(move || mark_inherited(&mut descriptors));
        }
        let error = command
            .spawn()
            .expect_err("truncated list must reject exec");
        assert_eq!(error.raw_os_error(), Some(libc::ENOBUFS));
    }
}
