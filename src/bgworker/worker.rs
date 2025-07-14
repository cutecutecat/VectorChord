use super::mq::MessageQueueWorker;
use super::utils::{Command, MAX_QUERY_LEN, Query, set_command, wait_command};
use crate::index::gucs::log_latest_queries;
use postcard::{from_bytes, to_vec};
use std::collections::VecDeque;

pub struct QueryLoggerMaster {
    mq_worker: MessageQueueWorker,
}

impl QueryLoggerMaster {
    pub unsafe fn new() -> Self {
        let mq_worker = unsafe { MessageQueueWorker::new() };
        Self { mq_worker }
    }

    pub unsafe fn push(&self, query: Query) {
        unsafe {
            let multi_reader_lock = super::mq::init_lock(super::mq::MULTI_READER_LOCK);
            let status = pgrx::pg_sys::LockAcquire(
                &multi_reader_lock as *const _ as *mut _,
                pgrx::pg_sys::ExclusiveLock as _,
                false,
                true,
            );
            if status == pgrx::pg_sys::LockAcquireResult::LOCKACQUIRE_NOT_AVAIL {
                pgrx::warning!("Failed to acquire lock for Push command");
                return;
            }
            match to_vec::<_, MAX_QUERY_LEN>(&query) {
                Ok(bytes) => match self.mq_worker.send(&bytes) {
                    Ok(_) => {}
                    Err(e) => {
                        pgrx::warning!("Failed to send Push command: {:?}", e);
                    }
                },
                Err(e) => {
                    pgrx::warning!("Failed to serialize Push command: {:?}", e);
                }
            };
            set_command(Command::Push);
            pgrx::pg_sys::LockRelease(
                &multi_reader_lock as *const _ as *mut _,
                pgrx::pg_sys::ExclusiveLock as _,
                false,
            );
        }
    }

    pub unsafe fn load_all(&self) -> Vec<Query> {
        unsafe {
            let multi_reader_lock = super::mq::init_lock(super::mq::MULTI_READER_LOCK);
            pgrx::pg_sys::LockAcquire(
                &multi_reader_lock as *const _ as *mut _,
                pgrx::pg_sys::AccessExclusiveLock as _,
                false,
                false,
            );
            set_command(Command::Length);
            let length = match self.mq_worker.recv() {
                Ok(bytes) => match from_bytes::<u32>(&bytes) {
                    Ok(l) => l,
                    Err(e) => {
                        pgrx::warning!("Error receiving message: {:?}", e);
                        0
                    }
                },
                Err(e) => {
                    pgrx::warning!("Error receiving message: {:?}", e);
                    0
                }
            };
            let mut queries = Vec::with_capacity(length as usize);
            for i in 0..length {
                set_command(Command::Load(i));
                match self.mq_worker.recv() {
                    Ok(bytes) => {
                        if let Ok(query) = from_bytes::<Query>(&bytes) {
                            queries.push(query);
                        } else {
                            pgrx::warning!("Failed to deserialize LoadOk response");
                        }
                    }
                    Err(e) => {
                        pgrx::warning!("Error receiving message: {:?}", e);
                        break;
                    }
                }
            }
            pgrx::pg_sys::LockRelease(
                &multi_reader_lock as *const _ as *mut _,
                pgrx::pg_sys::ExclusiveLock as _,
                false,
            );
            queries
        }
    }
}

pub struct QueryLoggerWorker {
    query_queue: VecDeque<Query>,
    mq_worker: MessageQueueWorker,
}

impl QueryLoggerWorker {
    pub unsafe fn new() -> Self {
        let mq_worker = unsafe { MessageQueueWorker::new() };
        Self {
            query_queue: VecDeque::new(),
            mq_worker,
        }
    }
    pub unsafe fn run(&mut self) {
        pgrx::warning!("BackgroundWorkerImpl started processing messages.");

        loop {
            pgrx::check_for_interrupts!();
            unsafe {
                if pgrx::pg_sys::ConfigReloadPending != 0 {
                    pgrx::warning!("Configuration reload pending, reloading...");
                    pgrx::pg_sys::ProcessConfigFile(pgrx::pg_sys::SIGHUP);
                }
            }
            let max_length = log_latest_queries();
            let command = unsafe { wait_command() };
            match command {
                Command::Push => {
                    let msg_bytes = if let Ok(Some(b)) = self.mq_worker.try_recv() {
                        b
                    } else {
                        pgrx::warning!("No message received for Push command");
                        continue;
                    };
                    let query = if let Ok(q) = from_bytes::<Query>(&msg_bytes) {
                        q
                    } else {
                        pgrx::warning!("Failed to deserialize query from message");
                        continue;
                    };
                    while self.query_queue.len() >= max_length as usize {
                        self.query_queue.pop_back();
                    }
                    self.query_queue.push_front(query);
                }
                Command::Length => {
                    let length = self.query_queue.len() as u32;
                    let msg_bytes = if let Ok(b) = to_vec::<_, MAX_QUERY_LEN>(&length) {
                        b
                    } else {
                        pgrx::warning!("Failed to serialize length response");
                        continue;
                    };
                    if let Err(e) = self.mq_worker.send(&msg_bytes) {
                        pgrx::warning!("Failed to send LengthOk response: {:?}", e);
                    }
                }
                Command::Load(index) => {
                    let query = if let Some(q) = self.query_queue.get(index as usize).cloned() {
                        q
                    } else {
                        pgrx::warning!("No query found at index {}", index);
                        continue;
                    };
                    let msg_bytes = if let Ok(b) = to_vec::<_, MAX_QUERY_LEN>(&query) {
                        b
                    } else {
                        pgrx::warning!("Failed to serialize length response");
                        continue;
                    };
                    if let Err(e) = self.mq_worker.send(&msg_bytes) {
                        pgrx::warning!("Failed to send LengthOk response: {:?}", e);
                    }
                }
                Command::Shutdown => {
                    pgrx::warning!("Shutdown requested, exiting worker loop.");
                    break;
                }
                Command::None => {}
            }
        }
    }
}
