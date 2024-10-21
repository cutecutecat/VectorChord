#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::identity_op)]

mod algorithm;
mod datatype;
mod gucs;
mod index;
mod postgres;
mod types;
mod upgrade;
mod utils;

pgrx::pg_module_magic!();
pgrx::extension_sql_file!("./sql/bootstrap.sql", bootstrap);
pgrx::extension_sql_file!("./sql/finalize.sql", finalize);

#[pgrx::pg_guard]
unsafe extern "C" fn _PG_init() {
    detect::init();
    unsafe {
        index::init();
        gucs::init();
    }
}

#[cfg(not(target_endian = "little"))]
compile_error!("Target architecture is not supported.");