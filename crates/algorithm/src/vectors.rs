use crate::operator::*;
use crate::tuples::*;
use crate::{IndexPointer, Page, PageGuard, RelationRead, RelationReadBatch, RelationWrite, tape};
use std::collections::BTreeMap;
use std::num::NonZero;
use vector::VectorOwned;

pub fn read_for_h1_tuple<
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    index: impl RelationRead,
    mean: IndexPointer,
    accessor: A,
) -> A::Output {
    let mut cursor = Err(mean);
    let mut result = accessor;
    while let Err(mean) = cursor.map_err(pointer_to_pair) {
        let guard = index.read(mean.0);
        let bytes = guard.get(mean.1).expect("data corruption");
        let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
        if tuple.payload().is_some() {
            panic!("data corruption");
        }
        result.push(tuple.elements());
        cursor = tuple.metadata_or_pointer();
    }
    result.finish(cursor.expect("data corruption"))
}

pub fn read_stream_for_h1_tuple<
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata> + Clone,
>(
    index: impl RelationReadBatch,
    means: Vec<IndexPointer>,
    accessor: A,
) -> Vec<A::Output> {
    let mut accessors: Vec<A> = vec![accessor; means.len()];
    let mut collect_outputs: BTreeMap<usize, A::Output> = BTreeMap::new();

    let mut means: Vec<_> = means
        .iter()
        .map(|ptr| Some(pointer_to_pair(*ptr)))
        .collect();
    loop {
        let block_ids: Vec<_> = means
            .iter()
            .filter_map(|opt| opt.as_ref().map(|o| o.0))
            .collect();

        let guards = index.read_batch(block_ids);

        let mut all_stream_end = true;
        let mut processed_index = 0;

        means = means
            .iter()
            .enumerate()
            .map(|(i, item)| {
                if let Some((_, buffer_id)) = item {
                    let bytes = guards[processed_index]
                        .get(*buffer_id)
                        .expect("data corruption");
                    processed_index += 1;
                    let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);

                    if tuple.payload().is_some() {
                        panic!("data corruption");
                    }
                    accessors[i].push(tuple.elements());

                    let cursor = tuple.metadata_or_pointer();
                    match cursor {
                        Ok(_) => {
                            collect_outputs.insert(
                                i,
                                accessors[i]
                                    .clone()
                                    .finish(cursor.expect("data corruption")),
                            );
                            None
                        }
                        Err(ptr) => {
                            all_stream_end = false;
                            Some(pointer_to_pair(ptr))
                        }
                    }
                } else {
                    None
                }
            })
            .collect();
        if all_stream_end {
            break;
        }
    }

    let mut result = vec![];
    for i in 0..means.len() {
        result.push(collect_outputs.get(&i).unwrap());
    }
    let mut result: Vec<_> = collect_outputs.into_iter().collect();
    result.sort_by_key(|item| item.0);
    result.into_iter().map(|(_, b)| b).collect()
}

pub fn read_for_h0_tuple<
    O: Operator,
    A: Accessor1<<O::Vector as Vector>::Element, <O::Vector as Vector>::Metadata>,
>(
    index: impl RelationRead,
    mean: IndexPointer,
    payload: NonZero<u64>,
    accessor: A,
) -> Option<A::Output> {
    let mut cursor = Err(mean);
    let mut result = accessor;
    while let Err(mean) = cursor.map_err(pointer_to_pair) {
        let guard = index.read(mean.0);
        let bytes = guard.get(mean.1)?;
        let tuple = VectorTuple::<O::Vector>::deserialize_ref(bytes);
        if tuple.payload().is_none() {
            panic!("data corruption");
        }
        if tuple.payload() != Some(payload) {
            return None;
        }
        result.push(tuple.elements());
        cursor = tuple.metadata_or_pointer();
    }
    Some(result.finish(cursor.ok()?))
}

pub fn append<O: Operator>(
    index: impl RelationWrite,
    vectors_first: u32,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    payload: NonZero<u64>,
) -> IndexPointer {
    fn append(index: impl RelationWrite, first: u32, bytes: &[u8]) -> IndexPointer {
        if let Some(mut write) = index.search(bytes.len()) {
            let i = write
                .alloc(bytes)
                .expect("implementation: a free page cannot accommodate a single tuple");
            return pair_to_pointer((write.id(), i));
        }
        tape::append(index, first, bytes, true)
    }
    let (slices, metadata) = O::Vector::split(vector);
    let mut chain = Ok(metadata);
    for i in (0..slices.len()).rev() {
        let bytes = VectorTuple::<O::Vector>::serialize(&match chain {
            Ok(metadata) => VectorTuple::_0 {
                elements: slices[i].to_vec(),
                payload: Some(payload),
                metadata,
            },
            Err(pointer) => VectorTuple::_1 {
                elements: slices[i].to_vec(),
                payload: Some(payload),
                pointer,
            },
        });
        chain = Err(append(index.clone(), vectors_first, &bytes));
    }
    chain.expect_err("internal error: 0-dimensional vector")
}
