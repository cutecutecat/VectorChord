use crate::operator::*;
use crate::tuples::{MetaTuple, WithReader};
use crate::{IndexPointer, Page, RelationRead, RelationReadBatch, RerankMethod, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::VectorOwned;

type Result<T> = (
    Reverse<Distance>,
    AlwaysEqual<T>,
    AlwaysEqual<NonZero<u64>>,
    AlwaysEqual<IndexPointer>,
);

type Rerank = (Reverse<Distance>, AlwaysEqual<NonZero<u64>>);

pub fn how(index: impl RelationRead) -> RerankMethod {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    if rerank_in_heap {
        RerankMethod::Heap
    } else {
        RerankMethod::Index
    }
}

pub struct Reranker<T, F> {
    heap: BinaryHeap<Result<T>>,
    cache: BinaryHeap<(Reverse<Distance>, AlwaysEqual<NonZero<u64>>)>,
    f: F,
    stream_length: u32,
}

impl<T, F: FnMut(&[IndexPointer], &[NonZero<u64>]) -> Vec<Option<Distance>>> Iterator
    for Reranker<T, F>
{
    type Item = (Distance, NonZero<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let elements = pop_n_if(&mut self.heap, self.stream_length, |(d, ..)| {
                Some(*d) > self.cache.peek().map(|(d, ..)| *d)
            });
            if elements.is_empty() {
                break;
            }
            let pay_u_stream: Vec<NonZero<u64>> = elements.iter().map(|m| m.2.0).collect();
            let mean_stream: Vec<IndexPointer> = elements.iter().map(|m| m.3.0).collect();
            let dis_u_collect: Vec<Option<Distance>> = (self.f)(&mean_stream, &pay_u_stream);
            for (dis_u_opt, pay_u) in dis_u_collect.iter().zip(pay_u_stream) {
                if let Some(dis_u) = dis_u_opt {
                    self.cache.push((Reverse(*dis_u), AlwaysEqual(pay_u)));
                }
            }
        }
        let (Reverse(dis_u), AlwaysEqual(pay_u)) = self.cache.pop()?;
        Some((dis_u, pay_u))
    }
}

impl<T, F> Reranker<T, F> {
    pub fn finish(
        self,
    ) -> (
        impl Iterator<Item = Result<T>>,
        impl Iterator<Item = Rerank>,
    ) {
        (self.heap.into_iter(), self.cache.into_iter())
    }
}

#[allow(clippy::type_complexity)]
pub fn rerank_index<O: Operator, T>(
    index: impl RelationReadBatch,
    vector: O::Vector,
    results: Vec<Result<T>>,
    stream_length: u32,
) -> Reranker<T, impl FnMut(&[IndexPointer], &[NonZero<u64>]) -> Vec<Option<Distance>>> {
    Reranker {
        heap: BinaryHeap::from(results),
        cache: BinaryHeap::<(Reverse<Distance>, _)>::new(),
        stream_length,
        f: move |mean_stream: &[IndexPointer], pay_u_stream: &[NonZero<u64>]| {
            vectors::read_stream_for_h0_tuple::<O, _>(
                index.clone(),
                mean_stream,
                pay_u_stream,
                LTryAccess::new(
                    O::Vector::unpack(vector.as_borrowed()),
                    O::DistanceAccessor::default(),
                ),
            )
        },
    }
}

#[allow(clippy::type_complexity)]
pub fn rerank_heap<O: Operator, T>(
    vector: O::Vector,
    results: Vec<Result<T>>,
    mut fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector>,
) -> Reranker<T, impl FnMut(&[IndexPointer], &[NonZero<u64>]) -> Vec<Option<Distance>>> {
    Reranker {
        heap: BinaryHeap::from(results),
        cache: BinaryHeap::<(Reverse<Distance>, _)>::new(),
        stream_length: 1,
        f: move |_: &[IndexPointer], pay_u_stream: &[NonZero<u64>]| {
            let vector = O::Vector::unpack(vector.as_borrowed());
            pay_u_stream
                .iter()
                .map(|pay_u| {
                    let vec_u = fetch(*pay_u)?;
                    let vec_u = O::Vector::unpack(vec_u.as_borrowed());
                    let mut accessor = O::DistanceAccessor::default();
                    accessor.push(vector.0, vec_u.0);
                    let dis_u = accessor.finish(vector.1, vec_u.1);
                    Some(dis_u)
                })
                .collect()
        },
    }
}

// fn pop_if<T: Ord>(
//     heap: &mut BinaryHeap<T>,
//     mut predicate: impl FnMut(&mut T) -> bool,
// ) -> Option<T> {
//     use std::collections::binary_heap::PeekMut;
//     let mut peek = heap.peek_mut()?;
//     if predicate(&mut peek) {
//         Some(PeekMut::pop(peek))
//     } else {
//         None
//     }
// }

fn pop_n_if<T: Ord>(
    heap: &mut BinaryHeap<T>,
    n: u32,
    mut predicate: impl FnMut(&mut T) -> bool,
) -> Vec<T> {
    use std::collections::binary_heap::PeekMut;
    let mut vec = vec![];
    for _ in 0..n {
        let peek = heap.peek_mut();
        match peek {
            Some(mut p) => {
                if predicate(&mut p) {
                    vec.push(PeekMut::pop(p));
                } else {
                    return vec;
                }
            }
            None => return vec,
        }
    }
    vec
}
