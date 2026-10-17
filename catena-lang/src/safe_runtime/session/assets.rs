//! Verified descriptor attachment and immutable-asset handoff.

use super::super::{
    AssetError, AttachedAsset, ResidentError, SafeRuntime,
    assets::{MAX_ASSET_BYTES, validate_byte_len},
    fd_transport::receive_file,
    protocol::{ProtocolError, Request, Response, WireAssetSlice},
    worker::{DeadlineKind, WorkerProcess, map_asset_worker_error, map_resident_worker_error},
};
use std::{fs::File, os::fd::AsRawFd};

impl SafeRuntime {
    pub(crate) fn attach_asset(
        &self,
        key: [u8; 32],
        file: File,
    ) -> Result<AttachedAsset, AssetError> {
        let byte_len = file.metadata().map_err(AssetError::FileMetadata)?.len();
        self.attach_asset_range(key, file, 0, byte_len)
    }

    pub(crate) fn attach_asset_range(
        &self,
        key: [u8; 32],
        file: File,
        offset: u64,
        byte_len: u64,
    ) -> Result<AttachedAsset, AssetError> {
        if let Some(owner) = &self.asset_owner {
            return owner.attach_asset_range(key, file, offset, byte_len);
        }
        let file_bytes = file.metadata().map_err(AssetError::FileMetadata)?.len();
        if validate_byte_len(byte_len).is_err() {
            return Err(AssetError::InvalidLength {
                actual: byte_len,
                maximum: MAX_ASSET_BYTES,
            });
        }
        if offset
            .checked_add(byte_len)
            .is_none_or(|end| end > file_bytes)
        {
            return Err(AssetError::InvalidSlice {
                offset,
                byte_len,
                asset_byte_len: file_bytes,
            });
        }

        let _operation = self.begin_operation().map_err(map_asset_worker_error)?;
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| AssetError::Transport("worker lock was poisoned".to_string()))?;
        if let Some(termination) = worker.termination() {
            return Err(AssetError::Unavailable {
                status: termination.status,
                stderr: termination.stderr.clone(),
            });
        }
        match worker
            .with_deadline(
                self.timeouts.execution_timeout(),
                DeadlineKind::Execution,
                |worker, deadline| {
                    let request = if offset == 0 && byte_len == file_bytes {
                        Request::AttachAsset { key, byte_len }
                    } else {
                        Request::AttachAssetRange {
                            key,
                            file_bytes,
                            offset,
                            byte_len,
                        }
                    };
                    worker.send(&request, deadline)?;
                    worker.send_file(&file)?;
                    worker.receive(deadline)
                },
            )
            .map_err(map_asset_worker_error)?
        {
            Response::Attached(Ok((id, reported_len))) if reported_len == byte_len => {
                Ok(AttachedAsset { id, byte_len })
            }
            Response::Attached(Ok(_)) => {
                Err(worker.reject_protocol_response(AssetError::UnexpectedResponse))
            }
            Response::Attached(Err(error)) => Err(AssetError::Remote(error)),
            _ => Err(worker.reject_protocol_response(AssetError::UnexpectedResponse)),
        }
    }

    pub(crate) fn asset_usage(&self) -> Result<(u64, u64, u64), AssetError> {
        if let Some(owner) = &self.asset_owner {
            return owner.asset_usage();
        }
        let _operation = self.begin_operation().map_err(map_asset_worker_error)?;
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| AssetError::Transport("worker lock was poisoned".into()))?;
        match worker
            .with_deadline(
                self.timeouts.execution_timeout(),
                DeadlineKind::Execution,
                |worker, deadline| worker.request(&Request::AssetUsage, deadline),
            )
            .map_err(map_asset_worker_error)?
        {
            Response::AssetUsage(count, uploaded, device) => Ok((count, uploaded, device)),
            _ => Err(worker.reject_protocol_response(AssetError::UnexpectedResponse)),
        }
    }

    pub(super) fn export_asset(&self, asset: u64) -> Result<(File, u64, u64, u64), ResidentError> {
        let _operation = self.begin_operation().map_err(map_resident_worker_error)?;
        let mut worker = self.resident_worker()?;
        worker
            .with_deadline(
                self.timeouts.execution_timeout(),
                DeadlineKind::Execution,
                |worker, deadline| {
                    let response = worker.request(&Request::ExportAsset { asset }, deadline)?;
                    let result = match response {
                        Response::Exported(Ok((byte_len, allocation_len, file_offset))) => {
                            deadline
                                .wait_for_descriptor(worker.asset_socket.as_raw_fd(), libc::POLLIN)
                                .map_err(ProtocolError::Io)?;
                            let file = receive_file(&worker.asset_socket)?;
                            Ok((file, byte_len, allocation_len, file_offset))
                        }
                        Response::Exported(Err(error)) => Err(ResidentError::Remote(error)),
                        _ => {
                            Err(worker.reject_protocol_response(ResidentError::UnexpectedResponse))
                        }
                    };
                    Ok(result)
                },
            )
            .map_err(map_resident_worker_error)?
    }

    pub(super) fn import_resident_assets(
        &self,
        worker: &mut WorkerProcess,
        slices: &[WireAssetSlice],
    ) -> Result<(), ResidentError> {
        for slice in slices {
            if worker.imported_assets.contains(&slice.asset) {
                continue;
            }
            let owner = self
                .asset_owner
                .as_ref()
                .ok_or_else(|| ResidentError::Remote("session has no asset owner".into()))?;
            let (file, byte_len, allocation_len, file_offset) = owner.export_asset(slice.asset)?;
            let response = worker
                .with_deadline(
                    self.timeouts.execution_timeout(),
                    DeadlineKind::Execution,
                    |worker, deadline| {
                        worker.send(
                            &Request::ImportAsset {
                                asset: slice.asset,
                                byte_len,
                                allocation_len,
                                file_offset,
                            },
                            deadline,
                        )?;
                        worker.send_file(&file)?;
                        worker.receive(deadline)
                    },
                )
                .map_err(map_resident_worker_error)?;
            match response {
                Response::Attached(Ok((id, len))) if id == slice.asset && len == byte_len => {
                    worker.imported_assets.insert(id);
                }
                Response::Attached(Err(error)) => return Err(ResidentError::Remote(error)),
                _ => return Err(worker.reject_protocol_response(ResidentError::UnexpectedResponse)),
            }
        }
        Ok(())
    }
}
