use pgrx::pg_sys::Oid;
use pgrx::pg_sys::panic::ErrorReportable;
use pgrx_catalog::{PgClass, PgNamespace};

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
use crate::collector::Operator;
use crate::collector::types::{
    BgWorkerLockGuard, Command, MAX_FLOATS_PER_INDEX, Query, VCHORD_MAGIC, WorkerState,
};
use crate::collector::unix::{accept, connect};
use crate::index::gucs;
use std::collections::{HashMap, VecDeque};
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, RwLock};
use std::time::Duration;

pub const MULTI_CONNECT_LOCK: u32 = 0;
pub static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);
const INTERNAL_TABLE_NAME: &str = "_internal_vchord_query_storage";
static INTERNAL_TABLE_SCHEMA: LazyLock<RwLock<String>> =
    LazyLock::new(|| RwLock::new(String::from("public")));

pub struct QueryCollectorMaster {}

impl QueryCollectorMaster {
    pub fn init() {
        pgrx::spi::Spi::connect_mut(|client| {
            let namespace_query = "SELECT n.nspname::TEXT
                FROM pg_catalog.pg_extension e
                LEFT JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
                WHERE e.extname = 'vchord';";
            let vchord_namespace: String = client
                .select(namespace_query, None, &[])
                .unwrap_or_report()
                .first()
                .get_by_name("nspname")
                .expect("external build: cannot get namespace of vchord")
                .expect("external build: cannot get namespace of vchord");
            let mut namespace_guard = INTERNAL_TABLE_SCHEMA.write().unwrap();
            *namespace_guard = vchord_namespace.clone();

            let namespace_oid = {
                let c_namespace = CString::new(vchord_namespace.as_str()).unwrap();
                let tuple = PgNamespace::search_namespacename(&c_namespace);
                if let Some(tuple) = tuple {
                    let pg_namespace = tuple.get().unwrap();
                    pg_namespace.oid()
                } else {
                    pgrx::warning!(
                        "Namespace with namespace {:?} not found in syscache",
                        c_namespace
                    );
                    return;
                }
            };
            let exist = {
                let c_table = CString::new(INTERNAL_TABLE_NAME).unwrap();
                let tuple = PgClass::search_relnamensp(&c_table, namespace_oid);
                if let Some(inner) = tuple
                    && inner.get().is_some()
                {
                    true
                } else {
                    false
                }
            };

            if !exist {
                let create_sql = format!(
                    "CREATE TABLE IF NOT EXISTS {vchord_namespace}.{INTERNAL_TABLE_NAME} (
                     id BIGSERIAL PRIMARY KEY, table_oid OID, index_oid OID,
                     operator TEXT, data TEXT)",
                );
                let _ = client.update(&create_sql, None, &[]);
            }
        });
    }

    pub unsafe fn push(query: Query) {
        QueryCollectorMaster::init();
        let namespace_guard = INTERNAL_TABLE_SCHEMA.read().unwrap();
        let namespace = &*namespace_guard;
        let Query {
            database_oid: _,
            table_oid,
            index_oid,
            operator,
            vector,
        } = query;
        let ops = operator.to_string();
        let vector_format: Vec<String> = vector.iter().map(|f| format!("{f:.2}")).collect();
        let joined_elements = vector_format.join(", ");
        let vector_text_rep = format!("[{joined_elements}]");
        pgrx::spi::Spi::connect_mut(|client| {
            let insert_sql = format!(
                "INSERT INTO {namespace}.{INTERNAL_TABLE_NAME} (table_oid, index_oid, operator, data) VALUES
                     ({table_oid}, {index_oid}, '{ops}', '{vector_text_rep}')",
            );
            let _ = client.update(&insert_sql, None, &[]);
        });
    }
    pub unsafe fn drop(_database_oid: u32, index_oid: u32) {
        QueryCollectorMaster::init();
        let namespace_guard = INTERNAL_TABLE_SCHEMA.read().unwrap();
        let namespace = &*namespace_guard;
        pgrx::spi::Spi::connect_mut(|client| {
            let insert_sql =
                format!("DELETE {namespace}.{INTERNAL_TABLE_NAME} WHERE index_oid = {index_oid}",);
            let _ = client.update(&insert_sql, None, &[]);
        });
    }
    pub unsafe fn load_all(_database_oid: u32, index_oid: u32) -> Vec<Query> {
        QueryCollectorMaster::init();
        let limit = gucs::vchordrq_max_logged_queries_per_index();
        let mut queries = Vec::new();
        let namespace_guard = INTERNAL_TABLE_SCHEMA.read().unwrap();
        let namespace = &*namespace_guard;

        let query_sql = format!(
            "SELECT table_oid, index_oid, operator, data
             FROM {namespace}.{INTERNAL_TABLE_NAME} WHERE index_oid = {index_oid} ORDER BY id DESC LIMIT {limit}",
        );

        pgrx::spi::Spi::connect(|client| {
            let rows = client
                .select(&query_sql, Some(limit as i64), &[])
                .unwrap_or_report();
            'r: for row in rows {
                let table_oid: Oid = if let Some(e) = row.get_by_name("table_oid").unwrap() {
                    e
                } else {
                    pgrx::warning!("table_oid is null in logged query");
                    continue 'r;
                };
                let index_oid: Oid = if let Some(e) = row.get_by_name("index_oid").unwrap() {
                    e
                } else {
                    pgrx::warning!("index_oid is null in logged query");
                    continue 'r;
                };
                let operator: &str = if let Some(e) = row.get_by_name("operator").unwrap() {
                    e
                } else {
                    pgrx::warning!("operator is null in logged query");
                    continue 'r;
                };
                let vector_text: &str = if let Some(e) = row.get_by_name("data").unwrap() {
                    e
                } else {
                    continue 'r;
                };
                let vector_text = vector_text.trim();
                let inner_str = &vector_text[1..vector_text.len() - 1];
                if inner_str.trim().is_empty() {
                    pgrx::warning!("Empty vector text, skipping");
                    continue 'r;
                }
                let parts = inner_str.split(',');

                let mut vector = Vec::new();
                for part in parts {
                    let trimmed_part = part.trim();
                    if trimmed_part.is_empty() {
                        pgrx::warning!("Empty part in vector text, skipping");
                        continue 'r;
                    }
                    match trimmed_part.parse::<f32>() {
                        Ok(value) => vector.push(value),
                        Err(e) => {
                            pgrx::warning!("Failed to parse vector part '{}': {}", trimmed_part, e);
                            continue 'r;
                        }
                    }
                }

                queries.push(Query {
                    database_oid: unsafe { pgrx::pg_sys::MyDatabaseId.to_u32() },
                    table_oid: table_oid.into(),
                    index_oid: index_oid.into(),
                    operator: Operator::try_from(operator).expect("Failed to parse operator"),
                    vector,
                });
            }
            queries
        })
    }
}

// pub struct QueryCollectorWorker {
//     db_index_query: HashMap<u32, HashMap<u32, VecDeque<Query>>>,
// }

// impl QueryCollectorWorker {
//     pub unsafe fn new() -> Self {
//         if let Some(restored) = WorkerState::load() {
//             pgrx::warning!(
//                 "Collector worker: Restored state with {} databases",
//                 restored.data.len()
//             );
//             Self {
//                 db_index_query: restored.data.clone(),
//             }
//         } else {
//             pgrx::warning!("Collector worker: No saved state found, starting fresh");
//             Self {
//                 db_index_query: HashMap::new(),
//             }
//         }
//     }
//     fn queue_length(&self, database_oid: u32, index_oid: u32) -> usize {
//         match self.db_index_query.get(&database_oid) {
//             Some(index_query) => index_query.get(&index_oid).map_or(0, |vec| vec.len()),
//             None => 0,
//         }
//     }
//     pub unsafe fn run(&mut self) {
//         pgrx::warning!("Collector worker: Starting query collector worker");
//         loop {
//             pgrx::check_for_interrupts!();
//             if SHUTDOWN_REQUESTED.load(Ordering::SeqCst) {
//                 pgrx::warning!("Collector worker: Received shutdown command");
//                 let saved = WorkerState {
//                     data: self.db_index_query.clone(),
//                     magic: VCHORD_MAGIC,
//                 };
//                 if let Err(e) = saved.save() {
//                     pgrx::warning!("Collector worker: Error saving state: {:?}", e);
//                 }
//                 break;
//             }
//             unsafe {
//                 if pgrx::pg_sys::ConfigReloadPending != 0 {
//                     pgrx::pg_sys::ConfigReloadPending = 0;
//                     pgrx::pg_sys::ProcessConfigFile(pgrx::pg_sys::GucContext::PGC_SIGHUP);
//                 }
//             }
//             pgrx::warning!("Collector: Waiting for connection");
//             let mut socket = match accept(Some(Duration::from_secs(1))) {
//                 Some(s) => s,
//                 None => continue,
//             };
//             pgrx::warning!("Collector: begin receiving query");
//             let command = socket.recv::<Command>().unwrap_or(Command::None);
//             match command {
//                 Command::Push(query) => {
//                     pgrx::warning!(
//                         "Collector worker: Received query: db={} table={} index={} opfamily={} vector_len={}",
//                         query.database_oid,
//                         query.table_oid,
//                         query.index_oid,
//                         query.operator,
//                         query.vector.len(),
//                     );
//                     let global_max_length = gucs::vchordrq_max_logged_queries_per_index();
//                     let local_max_length = (MAX_FLOATS_PER_INDEX / query.vector.len()) as u32;
//                     if local_max_length <= global_max_length {
//                         pgrx::warning!(
//                             "Collector worker: current vchordrq.max_logged_queries_per_index={} might be too high, better to set it to {}",
//                             global_max_length,
//                             local_max_length,
//                         );
//                     }
//                     let max_length = std::cmp::min(local_max_length, global_max_length);
//                     let database_oid = query.database_oid;
//                     let index_oid = query.index_oid;
//                     if max_length == 0 {
//                         self.db_index_query.entry(database_oid).and_modify(|inner| {
//                             inner.remove(&index_oid);
//                         });
//                     }
//                     while self.queue_length(database_oid, index_oid) >= max_length as usize
//                         && max_length > 0
//                     {
//                         pgrx::warning!(
//                             "Collector worker: Query queue is full, removing oldest query: {:?} > {}",
//                             self.queue_length(database_oid, index_oid),
//                             max_length
//                         );
//                         self.db_index_query.entry(database_oid).and_modify(|inner| {
//                             inner.entry(index_oid).and_modify(|e| {
//                                 e.pop_back();
//                             });
//                         });
//                     }
//                     self.db_index_query
//                         .entry(database_oid)
//                         .or_default()
//                         .entry(index_oid)
//                         .or_default()
//                         .push_front(query);
//                 }
//                 Command::Load(database_oid, index_oid) => {
//                     pgrx::warning!(
//                         "Collector worker: Load queries for db={} index={}",
//                         database_oid,
//                         index_oid
//                     );
//                     let result: Vec<Query> = self
//                         .db_index_query
//                         .get(&database_oid)
//                         .and_then(|inner_map| inner_map.get(&index_oid))
//                         .cloned()
//                         .unwrap_or_default()
//                         .into_iter()
//                         .collect();
//                     pgrx::warning!("Collector worker: Loaded {} queries", result.len());
//                     if let Err(e) = socket.send(result) {
//                         pgrx::warning!("Collector worker: Error sending query: {}", e);
//                     }
//                     pgrx::warning!("Collector worker: Sent queries");
//                 }
//                 Command::Drop(database_oid, index_oid) => {
//                     if let Some(inner) = self.db_index_query.get_mut(&database_oid)
//                         && let Some(queries) = inner.get_mut(&index_oid)
//                     {
//                         queries.clear();
//                     }
//                 }
//                 Command::Shutdown => {
//                     pgrx::warning!("Collector worker: Received shutdown command");
//                     let saved = WorkerState {
//                         data: self.db_index_query.clone(),
//                         magic: VCHORD_MAGIC,
//                     };
//                     if let Err(e) = saved.save() {
//                         pgrx::warning!("Collector worker: Error saving state: {:?}", e);
//                     }
//                     break;
//                 }
//                 Command::None => {}
//             }
//         }
//     }
// }
