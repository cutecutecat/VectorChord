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

use crate::square::Square;
use simd::Floating;
use std::collections::BinaryHeap;

pub fn approx_kmeans<F: FnMut() -> Option<(Vec<f32>, Vec<f32>)>>(
    kmeans_d: usize,
    original_d: usize,
    samples: Square,
    c: usize,
    num_threads: usize,
    seed: [u8; 32],
    is_spherical: bool,
    num_iterations: usize,
    mut next_sample: F,
) -> Square {
    let samples_len = samples.len();
    let top_list =
        ((c as f64).sqrt().floor() as u32).clamp(1, (samples_len as f64).sqrt().floor() as u32);
    let mut f = crate::k_means(kmeans_d, samples, top_list as usize, num_threads, seed);
    if is_spherical {
        f.sphericalize();
    }
    for _ in 0..num_iterations {
        f.assign();
        f.update();
        if is_spherical {
            f.sphericalize();
        }
    }
    let final_assign = f.assign();
    let (top_centroids, samples) = f.finish_samples();
    let alloc = final_assign.into_iter().enumerate().fold(
        vec![vec![]; top_centroids.len()],
        |mut acc, (i, target)| {
            acc[target].push(i);
            acc
        },
    );
    let alloc_size = alloc.iter().map(|x| x.len() as u32).collect::<Vec<_>>();
    let keep_indices: Vec<usize> = alloc_size
        .iter()
        .enumerate()
        .filter_map(|(i, size)| if *size > 0 { Some(i) } else { None })
        .collect();
    let alloc: Vec<_> = keep_indices.iter().map(|&i| alloc[i].clone()).collect();
    let alloc_size: Vec<_> = keep_indices.iter().map(|&i| alloc_size[i]).collect();
    let alloc_lists = successive_quotients_allocate(c, alloc_size);
    let mut offset = 0;
    let mut ret = Square::new(original_d);
    let mut bottom_centroids = vec![];
    let mut flatten_id_start = vec![];
    for (i, nlist) in alloc_lists.into_iter().enumerate() {
        let alloc_i = if let Some(a) = alloc.get(i) {
            a.clone()
        } else {
            unreachable!()
        };
        let sub_samples = {
            let mut s = Square::new(kmeans_d);
            for j in alloc_i {
                s.push_slice(&samples[j]);
            }
            s
        };
        let mut f = crate::k_means(kmeans_d, sub_samples, nlist as usize, num_threads, seed);
        if is_spherical {
            f.sphericalize();
        }
        for _ in 0..num_iterations {
            f.assign();
            f.update();
            if is_spherical {
                f.sphericalize();
            }
        }
        let final_assign = f.assign();
        let sub_centroids = f.finish();
        let sub_alloc =
            final_assign
                .into_iter()
                .fold(vec![0_usize; sub_centroids.len()], |mut acc, target| {
                    acc[target] += 1;
                    acc
                });
        let sub_centroids = {
            let it = sub_centroids
                .into_iter()
                .enumerate()
                .filter(|(j, _)| sub_alloc[*j] > 0)
                .map(|(_, c)| c);
            let mut s = Square::new(kmeans_d);
            for c in it {
                s.push_slice(c);
            }
            s
        };
        flatten_id_start.push(offset);
        offset += sub_centroids.len();
        if original_d == kmeans_d {
            for i in 0..sub_centroids.len() {
                ret.push_slice(&sub_centroids[i]);
            }
        } else {
            bottom_centroids.push(sub_centroids);
        }
    }
    if original_d == kmeans_d {
        return ret;
    }
    let c = offset;
    let mut centroids = vec![vec![0.0_f32; original_d]; c];
    let mut count = vec![0_u32; c];
    let mut traveled = 0;
    while let Some((original, reduction)) = next_sample()
        && traveled < samples_len
    {
        let top_id = k_means_lookup(&reduction, &top_centroids);
        let bottom_id = k_means_lookup(&reduction, &bottom_centroids[top_id]);
        let cid = flatten_id_start[top_id] + bottom_id;
        centroids[cid] = f32::vector_add(&centroids[cid], &original);
        count[cid] += 1;
        traveled += 1;
    }
    (centroids, count) = centroids
        .into_iter()
        .zip(count)
        .filter(|(_, cnt)| *cnt > 0)
        .unzip();
    for i in 0..count.len() {
        k_means_centroids_inplace(&mut centroids[i], count[i], is_spherical);
        ret.push_slice(&centroids[i]);
    }
    ret
}

pub fn k_means_lookup(vector: &[f32], centroids: &Square) -> usize {
    assert_ne!(centroids.len(), 0);
    let mut result = (f32::INFINITY, 0);
    for i in 0..centroids.len() {
        let dis = f32::reduce_sum_of_d2(vector, &centroids[i]);
        if dis <= result.0 {
            result = (dis, i);
        }
    }
    result.1
}

fn k_means_centroids_inplace(centroids: &mut [f32], count: u32, is_spherical: bool) {
    assert!(!centroids.is_empty());
    assert!(count > 0);
    let dim = centroids.len();
    let c = 1.0 / count as f32;
    for d in 0..dim {
        centroids[d] *= c;
    }
    if is_spherical {
        let l = f32::reduce_sum_of_x2(centroids).sqrt();
        f32::vector_mul_scalar_inplace(centroids, 1.0 / l);
    }
}

/// Allocate clusters to different parts according to the given proportions
/// by successive quotients method.
///
/// See: https://en.wikipedia.org/wiki/Sainte-Lagu%C3%AB_method
fn successive_quotients_allocate(all_clusters: usize, proportion: Vec<u32>) -> Vec<u32> {
    let mut alloc_lists = vec![1u32; proportion.len()];
    let mut diff = all_clusters as i32 - proportion.len() as i32;
    if diff < 0 {
        panic!(
            "build.lists is too large: requested {}, but only {} are available.",
            all_clusters,
            proportion.len()
        );
    }
    let mut priorities: BinaryHeap<PriorityItem> = proportion
        .iter()
        .enumerate()
        .map(|(i, x)| PriorityItem {
            index: i,
            priority: *x as f64 / (alloc_lists[0] as f64),
        })
        .collect();
    while diff > 0 {
        let top = priorities.pop().unwrap();
        alloc_lists[top.index] += 1;
        priorities.push(PriorityItem {
            index: top.index,
            priority: proportion[top.index] as f64 / (alloc_lists[top.index] as f64),
        });
        diff -= 1;
    }
    alloc_lists
}

#[derive(Debug, PartialEq)]
struct PriorityItem {
    index: usize,
    priority: f64,
}

impl Eq for PriorityItem {}

impl PartialOrd for PriorityItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.priority.is_nan() {
            std::cmp::Ordering::Less
        } else if other.priority.is_nan() {
            std::cmp::Ordering::Greater
        } else {
            match self.priority.partial_cmp(&other.priority) {
                Some(ordering) => ordering,
                None => std::cmp::Ordering::Equal,
            }
        }
    }
}
