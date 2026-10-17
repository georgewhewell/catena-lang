//! One absolute deadline for watchdogs and nonblocking protocol I/O.

use super::lifecycle::ManagedProcessGroup;
use std::{
    io::{self, Read, Write},
    os::fd::RawFd,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// A nonblocking protocol reader that waits only until the operation's one
/// absolute deadline. Descendants retaining a killed worker's pipe therefore
/// cannot strand the caller in `read(2)` after the watchdog fires.
pub(super) struct DeadlineReader<'a, R> {
    inner: &'a mut R,
    descriptor: RawFd,
    deadline: &'a ProcessGroupDeadline,
}

impl<'a, R> DeadlineReader<'a, R> {
    pub(super) fn new(
        inner: &'a mut R,
        descriptor: RawFd,
        deadline: &'a ProcessGroupDeadline,
    ) -> Self {
        Self {
            inner,
            descriptor,
            deadline,
        }
    }
}

impl<R: Read> Read for DeadlineReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            self.deadline.ensure_io_time_remaining()?;
            match self.inner.read(buffer) {
                Ok(read) => return Ok(read),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    self.deadline
                        .wait_for_descriptor(self.descriptor, libc::POLLIN)?;
                }
                Err(error) => return Err(error),
            }
        }
    }
}

/// The write-side counterpart to [`DeadlineReader`]. `write_frame` may make
/// partial progress on a full pipe, but every retry consumes the same deadline.
pub(super) struct DeadlineWriter<'a, W> {
    inner: &'a mut W,
    descriptor: RawFd,
    deadline: &'a ProcessGroupDeadline,
}

impl<'a, W> DeadlineWriter<'a, W> {
    pub(super) fn new(
        inner: &'a mut W,
        descriptor: RawFd,
        deadline: &'a ProcessGroupDeadline,
    ) -> Self {
        Self {
            inner,
            descriptor,
            deadline,
        }
    }
}

impl<W: Write> Write for DeadlineWriter<'_, W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            self.deadline.ensure_io_time_remaining()?;
            match self.inner.write(buffer) {
                Ok(written) => return Ok(written),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    self.deadline
                        .wait_for_descriptor(self.descriptor, libc::POLLOUT)?;
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        loop {
            self.deadline.ensure_io_time_remaining()?;
            match self.inner.flush() {
                Ok(()) => return Ok(()),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    self.deadline
                        .wait_for_descriptor(self.descriptor, libc::POLLOUT)?;
                }
                Err(error) => return Err(error),
            }
        }
    }
}

pub(super) fn set_nonblocking(descriptor: RawFd) -> io::Result<()> {
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
    if flags == -1 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags | libc::O_NONBLOCK) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[derive(Debug)]
pub(super) struct ProcessGroupDeadline {
    cancel: Option<mpsc::Sender<()>>,
    watchdog: Option<JoinHandle<()>>,
    timed_out: Arc<AtomicBool>,
    started: Instant,
    timeout: Duration,
}

impl ProcessGroupDeadline {
    #[cfg(test)]
    pub(super) fn watchdog_fired(&self) -> bool {
        self.timed_out.load(Ordering::Acquire)
    }

    pub(super) fn arm(
        process_group: Arc<ManagedProcessGroup>,
        timeout: Duration,
    ) -> io::Result<Self> {
        let started = Instant::now();
        let (cancel, cancelled) = mpsc::channel();
        let timed_out = Arc::new(AtomicBool::new(false));
        let watchdog_timed_out = timed_out.clone();
        let watchdog_process_group = process_group;
        let watchdog = thread::Builder::new()
            .name("catena-safe-runtime-watchdog".to_string())
            .spawn(move || match cancelled.recv_timeout(timeout) {
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    watchdog_timed_out.store(true, Ordering::Release);
                    watchdog_process_group.kill();
                }
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {}
            })?;
        Ok(Self {
            cancel: Some(cancel),
            watchdog: Some(watchdog),
            timed_out,
            started,
            timeout,
        })
    }

    pub(super) fn timed_out(&self) -> bool {
        self.timed_out.load(Ordering::Acquire) || self.started.elapsed() >= self.timeout
    }

    pub(super) fn remaining(&self) -> Option<Duration> {
        if self.timed_out.load(Ordering::Acquire) {
            return None;
        }
        self.timeout.checked_sub(self.started.elapsed())
    }

    pub(super) fn ensure_io_time_remaining(&self) -> io::Result<()> {
        if self.remaining().is_none() {
            self.timed_out.store(true, Ordering::Release);
            Err(protocol_timeout_error())
        } else {
            Ok(())
        }
    }

    pub(super) fn wait_for_descriptor(
        &self,
        descriptor: RawFd,
        events: libc::c_short,
    ) -> io::Result<()> {
        loop {
            let Some(remaining) = self.remaining() else {
                self.timed_out.store(true, Ordering::Release);
                return Err(protocol_timeout_error());
            };
            let fractional_millisecond = remaining.subsec_nanos() % 1_000_000 != 0;
            let timeout_millis = remaining
                .as_millis()
                .saturating_add(u128::from(fractional_millisecond))
                .clamp(1, i32::MAX as u128) as i32;
            let mut descriptor = libc::pollfd {
                fd: descriptor,
                events,
                revents: 0,
            };
            let result = unsafe { libc::poll(&mut descriptor, 1, timeout_millis) };
            if result > 0 {
                if descriptor.revents & libc::POLLNVAL != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "SafeRuntime protocol descriptor became invalid",
                    ));
                }
                // POLLHUP/POLLERR are deliberately treated as readiness: the
                // following read/write reports EOF or the precise pipe error.
                return Ok(());
            }
            if result == 0 {
                continue;
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }

    pub(super) fn finish(mut self) -> bool {
        let elapsed = self.timed_out();
        self.cancel();
        elapsed || self.timed_out.load(Ordering::Acquire)
    }

    pub(super) fn cancel(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
        if let Some(watchdog) = self.watchdog.take()
            && watchdog.join().is_err()
        {
            self.timed_out.store(true, Ordering::Release);
        }
    }
}

pub(super) fn protocol_timeout_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::TimedOut,
        "SafeRuntime protocol deadline elapsed",
    )
}

impl Drop for ProcessGroupDeadline {
    fn drop(&mut self) {
        self.cancel();
    }
}
