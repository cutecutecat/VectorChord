use serde::{Deserialize, Serialize};
use vchordrq::types::OwnedVector;

use crate::index::vchordrq::opclass::Opfamily;

unsafe extern "C" {
    pub unsafe static mut QUERY_LOGGER_MQ: *mut pgrx::pg_sys::shm_mq;
    pub unsafe static mut COMMAND_REQUEST: *mut Command;
}

pub const MAX_QUERY_LEN: usize = 256000; // 256 KB

#[repr(C, align(8))]
#[derive(Debug, Clone)]
pub enum Command {
    None,
    Shutdown,
    Push,
    Length,
    Load(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operator {
    L2,
    Cosine,
    Dot,
}

impl Operator {
    fn as_text(&self) -> String {
        match self {
            Operator::L2 => "<->".to_string(),
            Operator::Cosine => "<=>".to_string(),
            Operator::Dot => "<#>".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueryDump {
    pub table_name: String,
    pub column_name: String,
    pub operator: String,
    pub vector_text: String,
    pub simplified_query: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub table_oid: u32,
    pub column_attno: i16,
    pub operator: Operator,
    pub vector: Vec<f32>,
}

impl Query {
    pub fn new(
        table_oid: u32,
        column_attno: Option<i16>,
        opfamily: Opfamily,
        vector: OwnedVector,
    ) -> Option<Self> {
        let operator = match opfamily {
            Opfamily::HalfvecCosine | Opfamily::VectorCosine => Operator::Cosine,
            Opfamily::HalfvecIp | Opfamily::VectorIp => Operator::Dot,
            Opfamily::HalfvecL2 | Opfamily::VectorL2 => Operator::L2,
            Opfamily::VectorMaxsim | Opfamily::HalfvecMaxsim => return None,
        };
        let vector = match vector {
            OwnedVector::Vecf32(v) => v.into_vec(),
            OwnedVector::Vecf16(v) => v.into_vec().into_iter().map(|f| f.to_f32()).collect(),
        };
        let column_attno = column_attno?;
        Some(Self {
            table_oid,
            column_attno,
            operator,
            vector,
        })
    }

    pub fn dump(&self) -> Option<QueryDump> {
        let table_name = unsafe {
            let tuple = pgrx::pg_sys::SearchSysCache1(
                pgrx::pg_sys::SysCacheIdentifier::RELOID as i32,
                pgrx::pg_sys::ObjectIdGetDatum(self.table_oid.into()),
            );
            pgrx::warning!("SearchSysCache finish {}", tuple.is_null());
            if tuple.is_null() {
                pgrx::warning!(
                    "Attribute with relid {} not found in syscache",
                    self.table_oid,
                );
                return None;
            }
            pgrx::warning!("Check finish");
            let mut is_null = false;
            pgrx::warning!("Begin get");
            let datum = pgrx::pg_sys::SysCacheGetAttr(
                pgrx::pg_sys::SysCacheIdentifier::RELOID as i32,
                tuple,
                pgrx::pg_sys::Anum_pg_class_relname as i16,
                &mut is_null,
            );
            pgrx::warning!("Finish get isNULL {is_null}");
            let inner = pgrx::pg_sys::DatumGetName(datum);
            let bytes: &[u8] = std::slice::from_raw_parts(
                (*inner).data.as_ptr() as *const u8,
                (*inner).data.len(),
            );
            let c_str =
                std::ffi::CStr::from_bytes_until_nul(bytes).expect("Slice contains null byte");
            c_str.to_str().map(|s| s.to_owned()).unwrap()
        };
        let column_name = unsafe {
            let tuple = pgrx::pg_sys::SearchSysCache2(
                pgrx::pg_sys::SysCacheIdentifier::ATTNUM as i32,
                pgrx::pg_sys::ObjectIdGetDatum(self.table_oid.into()),
                pgrx::pg_sys::Int16GetDatum(self.column_attno),
            );
            if tuple.is_null() {
                pgrx::warning!(
                    "Attribute with relid {} and attno {} not found in syscache",
                    self.table_oid,
                    self.column_attno
                );
                return None;
            }
            let mut is_null = false;
            let datum = pgrx::pg_sys::SysCacheGetAttr(
                pgrx::pg_sys::SysCacheIdentifier::ATTNUM as i32,
                tuple,
                pgrx::pg_sys::Anum_pg_attribute_attname as i16,
                &mut is_null,
            );
            let inner = pgrx::pg_sys::DatumGetName(datum);
            let bytes: &[u8] = std::slice::from_raw_parts(
                (*inner).data.as_ptr() as *const u8,
                (*inner).data.len(),
            );
            let c_str =
                std::ffi::CStr::from_bytes_until_nul(bytes).expect("Slice contains null byte");
            c_str.to_str().map(|s| s.to_owned()).unwrap()
        };
        pgrx::warning!("column name {column_name}");
        let operator = self.operator.as_text();
        let vector_format: Vec<String> = self.vector.iter().map(|f| format!("{f:.2}")).collect();
        let joined_elements = vector_format.join(", ");
        let vector_text = format!("'[{joined_elements}]'");
        let simplified_query =
            format!("SELECT {column_name} from {table_name} ORDER BY {vector_text}");
        Some(QueryDump {
            table_name,
            column_name,
            operator,
            vector_text,
            simplified_query,
        })
    }
}

pub fn startup_hook() {
    let mut found = false;

    unsafe fn estimate_size() -> pgrx::pg_sys::Size {
        let mut estimator = pgrx::pg_sys::shm_toc_estimator {
            space_for_chunks: MAX_QUERY_LEN + size_of::<Option<Command>>(),
            number_of_keys: 2,
        };
        unsafe { pgrx::pg_sys::shm_toc_estimate(&mut estimator) }
    }
    unsafe {
        let pgws = pgrx::pg_sys::ShmemInitStruct(
            c"vchord_query_logger".as_ptr(),
            estimate_size(),
            &mut found,
        );
        if !found {
            let toc = pgrx::pg_sys::shm_toc_create(super::mq::VCHORD_MAGIC, pgws, MAX_QUERY_LEN);
            QUERY_LOGGER_MQ = pgrx::pg_sys::shm_toc_allocate(toc, MAX_QUERY_LEN) as *mut _;
            pgrx::pg_sys::shm_toc_insert(toc, 0, QUERY_LOGGER_MQ.cast());

            COMMAND_REQUEST =
                pgrx::pg_sys::shm_toc_allocate(toc, size_of::<Option<Command>>())
                    as *mut _;
            pgrx::pg_sys::shm_toc_insert(toc, 0, COMMAND_REQUEST.cast());
        } else {
            let toc = pgrx::pg_sys::shm_toc_attach(super::mq::VCHORD_MAGIC, pgws);
            QUERY_LOGGER_MQ = pgrx::pg_sys::shm_toc_lookup(toc, 0, false) as *mut _;
        }
    }
}

pub unsafe extern "C-unwind" fn handle_sigterm(_postgres_signal_arg: i32) {
    unsafe { set_command(Command::Shutdown) };
}

pub unsafe fn set_command(command: Command) {
    unsafe {
        *COMMAND_REQUEST = command;
        if !pgrx::pg_sys::MyProc.is_null() {
            pgrx::pg_sys::SetLatch(pgrx::pg_sys::MyLatch);
        }
    }
}

pub unsafe fn wait_command() -> Command {
    unsafe {
        pgrx::pg_sys::WaitLatch(
            pgrx::pg_sys::MyLatch,
            (pgrx::pg_sys::WL_LATCH_SET
                | pgrx::pg_sys::WL_TIMEOUT
                | pgrx::pg_sys::WL_EXIT_ON_PM_DEATH) as _,
            1000,
            pgrx::pg_sys::WaitEventTimeout::WAIT_EVENT_PG_SLEEP,
        );
        pgrx::pg_sys::ResetLatch(pgrx::pg_sys::MyLatch);
        COMMAND_REQUEST.as_ref().cloned().unwrap_or(Command::None)
    }
}
