// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

pub use utils::{Query, startup_hook};
pub use worker::{QueryLoggerMaster, QueryLoggerWorker};

mod mq;
mod utils;
mod worker;

pub unsafe fn init() {
    use pgrx::bgworkers::BackgroundWorkerBuilder;
    use pgrx::bgworkers::BgWorkerStartTime;
    use std::time::Duration;
    BackgroundWorkerBuilder::new("vectors")
        .set_library("vchord")
        .set_function("_query_logger_main")
        .set_argument(None)
        .enable_shmem_access(None)
        .set_start_time(BgWorkerStartTime::PostmasterStart)
        .set_restart_time(Some(Duration::from_secs(15)))
        .load();
}

#[pgrx::pg_guard]
#[unsafe(no_mangle)]
pub extern "C-unwind" fn _query_logger_main(_arg: pgrx::pg_sys::Datum) {
    use core::mem::transmute;
    use pgrx::pg_sys;

    pgrx::warning!("my_background_worker_main entry point reached.");
    unsafe {
        pg_sys::pqsignal(pg_sys::SIGTERM as i32, Some(utils::handle_sigterm));
        pg_sys::pqsignal(
            pg_sys::SIGHUP as i32,
            transmute::<*mut (), pg_sys::pqsigfunc>(pg_sys::SignalHandlerForConfigReload as _),
        );
        pg_sys::pqsignal(
            pg_sys::SIGUSR1 as i32,
            transmute::<*mut (), pg_sys::pqsigfunc>(pg_sys::procsignal_sigusr1_handler as _),
        );

        let mut worker = QueryLoggerWorker::new();
        worker.run();
    }
}
