use super::utils::QUERY_LOGGER_MQ;
use pgrx::pg_sys;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ptr::NonNull;

pub const VCHORD_MAGIC: u64 = 0x5643484f;
pub const MULTI_READER_LOCK: u32 = 0;
pub const READER_LOGGER_LOCK: u32 = 1;

#[derive(Debug, Copy, Clone)]
pub enum MessageQueueError {
    Detached,
    WouldBlock,
    Unknown(pg_sys::shm_mq_result::Type),
}

impl Display for MessageQueueError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageQueueError::Detached => write!(f, "queue is detached"),
            MessageQueueError::WouldBlock => write!(f, "queue is full"),
            MessageQueueError::Unknown(other) => write!(f, "unknown error code: {other}"),
        }
    }
}

impl Error for MessageQueueError {}

impl From<pg_sys::shm_mq_result::Type> for MessageQueueError {
    fn from(value: pg_sys::shm_mq_result::Type) -> Self {
        match value {
            pg_sys::shm_mq_result::SHM_MQ_WOULD_BLOCK => Self::WouldBlock,
            pg_sys::shm_mq_result::SHM_MQ_DETACHED => Self::Detached,
            other => Self::Unknown(other),
        }
    }
}

pub struct MessageQueueWorker {
    handle: NonNull<pg_sys::shm_mq_handle>,
    send_lock: pg_sys::LOCKTAG,
}

impl Drop for MessageQueueWorker {
    fn drop(&mut self) {
        unsafe {
            if pg_sys::IsInParallelMode() {
                pg_sys::shm_mq_detach(self.handle.as_ptr());
            }
        }
    }
}

impl MessageQueueWorker {
    pub unsafe fn new() -> Self {
        unsafe {
            pg_sys::shm_mq_set_sender(QUERY_LOGGER_MQ, pg_sys::MyProc);
            pg_sys::shm_mq_set_receiver(QUERY_LOGGER_MQ, pg_sys::MyProc);
            let handle =
                pg_sys::shm_mq_attach(QUERY_LOGGER_MQ, std::ptr::null_mut(), std::ptr::null_mut());
            Self {
                handle: NonNull::new_unchecked(handle),
                send_lock: init_lock(READER_LOGGER_LOCK),
            }
        }
    }

    pub fn send<B: AsRef<[u8]>>(&self, msg: B) -> Result<(), MessageQueueError> {
        unsafe {
            pg_sys::LockAcquire(
                &self.send_lock as *const _ as *mut _,
                pg_sys::ExclusiveLock as _,
                false,
                false,
            );
            let msg = msg.as_ref();
            #[cfg(feature = "pg14")]
            let result = pg_sys::shm_mq_send(
                self.handle.as_ptr(),
                msg.len(),
                msg.as_ptr() as *mut std::ffi::c_void,
                false,
            );

            #[cfg(not(feature = "pg14"))]
            let result = pg_sys::shm_mq_send(
                self.handle.as_ptr(),
                msg.len(),
                msg.as_ptr() as *mut std::ffi::c_void,
                false,
                true,
            );
            pg_sys::LockRelease(
                &self.send_lock as *const _ as *mut _,
                pg_sys::ExclusiveLock as _,
                false,
            );

            match result {
                pg_sys::shm_mq_result::SHM_MQ_SUCCESS => Ok(()),
                other => Err(MessageQueueError::from(other)),
            }
        }
    }

    #[allow(dead_code)]
    pub fn try_send(&self, msg: &[u8]) -> Result<Option<()>, MessageQueueError> {
        unsafe {
            pg_sys::LockAcquire(
                &self.send_lock as *const _ as *mut _,
                pg_sys::ExclusiveLock as _,
                false,
                false,
            );
            #[cfg(feature = "pg14")]
            let result = pg_sys::shm_mq_send(
                self.handle.as_ptr(),
                msg.len(),
                msg.as_ptr() as *mut std::ffi::c_void,
                true,
            );

            #[cfg(not(feature = "pg14"))]
            let result = pg_sys::shm_mq_send(
                self.handle.as_ptr(),
                msg.len(),
                msg.as_ptr() as *mut std::ffi::c_void,
                true,
                true,
            );
            pg_sys::LockRelease(
                &self.send_lock as *const _ as *mut _,
                pg_sys::ExclusiveLock as _,
                false,
            );
            match result {
                pg_sys::shm_mq_result::SHM_MQ_SUCCESS => Ok(Some(())),
                pg_sys::shm_mq_result::SHM_MQ_WOULD_BLOCK => Ok(None),
                other => Err(MessageQueueError::from(other)),
            }
        }
    }

    pub fn recv(&self) -> Result<Vec<u8>, MessageQueueError> {
        unsafe {
            let mut len = 0usize;
            let mut msg = std::ptr::null_mut();
            let result = pg_sys::shm_mq_receive(self.handle.as_ptr(), &mut len, &mut msg, false);

            match result {
                pg_sys::shm_mq_result::SHM_MQ_SUCCESS => {
                    Ok(std::slice::from_raw_parts(msg as *mut u8, len).to_vec())
                }
                other => Err(MessageQueueError::from(other)),
            }
        }
    }

    pub fn try_recv(&self) -> Result<Option<Vec<u8>>, MessageQueueError> {
        unsafe {
            let mut len = 0usize;
            let mut msg = std::ptr::null_mut();
            let result = pg_sys::shm_mq_receive(self.handle.as_ptr(), &mut len, &mut msg, true);

            match result {
                pg_sys::shm_mq_result::SHM_MQ_SUCCESS => Ok(Some(
                    std::slice::from_raw_parts(msg as *mut u8, len).to_vec(),
                )),
                pg_sys::shm_mq_result::SHM_MQ_WOULD_BLOCK => Ok(None),
                other => Err(MessageQueueError::from(other)),
            }
        }
    }
}

pub fn init_lock(lock_id: u32) -> pg_sys::LOCKTAG {
    pg_sys::LOCKTAG {
        locktag_type: pg_sys::LockTagType::LOCKTAG_USERLOCK as u8,
        locktag_lockmethodid: pg_sys::USER_LOCKMETHOD as u8,
        locktag_field1: VCHORD_MAGIC as u32,
        locktag_field2: lock_id,
        locktag_field3: 0,
        locktag_field4: 0,
    }
}
