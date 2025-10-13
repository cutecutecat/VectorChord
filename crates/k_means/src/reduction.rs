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

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use simd::Floating;

pub struct KMeansReduction {
    pub projector: Vec<Vec<f32>>,
    pub inv_projector: Vec<Vec<f32>>,
    pub cholesky_l: Vec<Vec<f32>>, // Lower triangular matrix from Cholesky decomposition
    pub in_dims: u32,
    pub out_dims: u32,
}

impl KMeansReduction {
    pub fn new(in_dims: u32, out_dims: u32) -> Self {
        assert!(in_dims >= out_dims);
        let mut rng = StdRng::from_seed([7; 32]);
        let projector = (0..out_dims)
            .map(|_| {
                (0..in_dims)
                    .map(|_| rng.sample(StandardNormal))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        let inv_projector = (0..in_dims)
            .map(|i| {
                (0..out_dims)
                    .map(|j| projector[j as usize][i as usize])
                    .collect()
            })
            .collect();
        let mut pxt = vec![vec![0.0; out_dims as usize]; out_dims as usize];
        for i in 0..out_dims as usize {
            for j in 0..out_dims as usize {
                pxt[i][j] = f32::reduce_sum_of_xy(&projector[i], &projector[j]);
            }
        }
        let cholesky_l = cholesky_with_shift(&pxt);
        Self {
            projector,
            inv_projector,
            cholesky_l,
            in_dims,
            out_dims,
        }
    }
    pub fn project_vector(&self, input: Vec<f32>) -> Vec<f32> {
        assert_eq!(input.len() as u32, self.in_dims);
        (0..self.out_dims as usize)
            .map(|i| f32::reduce_sum_of_xy(&input, &self.projector[i]))
            .collect()
    }
    pub fn recover_centroids(&self, centroids: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        if centroids.is_empty() {
            return vec![];
        }
        assert_eq!(centroids[0].len() as u32, self.out_dims);
        let z = solve_spd_from_cholesky(self.cholesky_l.clone(), centroids);
        z.into_iter()
            .map(|s| {
                (0..self.in_dims as usize)
                    .map(|i| f32::reduce_sum_of_xy(&self.inv_projector[i], &s))
                    .collect()
            })
            .collect()
    }
}

fn cholesky_with_shift(m: &[Vec<f32>]) -> Vec<Vec<f32>> {
    const TRIES: u32 = 10;
    const EPSILON: f32 = 1e-12;
    let mut lambda = 1e-10;
    let n = m.len();
    assert!(n > 0 && m[0].len() == n);
    for _ in 0..TRIES {
        let mut cholesky_l = m.to_owned();
        for i in 0..n {
            cholesky_l[i][i] += lambda;
        }
        let mut ok = true;
        for i in 0..n {
            for j in 0..=i {
                let mut s = cholesky_l[i][j];
                for k in 0..j {
                    s -= cholesky_l[i][k] * cholesky_l[j][k];
                }
                if i == j {
                    if s <= EPSILON {
                        ok = false;
                        break;
                    }
                    cholesky_l[i][j] = s.sqrt();
                } else {
                    cholesky_l[i][j] = s / cholesky_l[j][j];
                }
            }
            if !ok {
                break;
            }
            for j in (i + 1)..n {
                cholesky_l[i][j] = 0.0;
            }
        }
        if ok {
            return cholesky_l;
        }
        lambda *= 10.0;
    }
    panic!("Cholesky failed even after diagonal shift");
}

fn solve_spd_from_cholesky(cholesky_l: Vec<Vec<f32>>, reduced: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut target = reduced.clone();
    let n = cholesky_l.len();
    assert!(n > 0 && cholesky_l[0].len() == n);
    let r = target.len();
    assert_eq!(target[0].len(), n);
    for i in 0..n {
        let lii = cholesky_l[i][i];
        assert!(lii > 0.0);
        for c in 0..r {
            let mut s = target[c][i];
            for k in 0..i {
                s -= cholesky_l[i][k] * target[c][k];
            }
            target[c][i] = s / lii;
        }
    }
    for i in (0..n).rev() {
        let lii = cholesky_l[i][i];
        for c in 0..r {
            let mut s = target[c][i];
            for k in (i + 1)..n {
                s -= cholesky_l[k][i] * target[c][k];
            }
            target[c][i] = s / lii;
        }
    }
    target
}

#[test]
fn precision_maintenance() {
    use rand_distr::{Distribution, StandardNormal, Uniform};
    use std::iter::zip;
    const TRIES: u32 = 1000;
    const DIM: u32 = 768;
    const EPSILON: f32 = 0.5;
    let reduction = KMeansReduction::new(DIM, 512);
    let mut rng = StdRng::from_os_rng();
    let sampler_1 = Uniform::try_from(-1.0..1.0).unwrap();
    for _ in 0..TRIES {
        let samples = vec![
            (0..DIM)
                .map(|_| rng.sample(StandardNormal))
                .collect::<Vec<f32>>(),
            (0..DIM)
                .map(|_| sampler_1.sample(&mut rng))
                .collect::<Vec<f32>>(),
            vec![sampler_1.sample(&mut rng); DIM as usize],
            vec![0.0; DIM as usize],
            vec![1.0; DIM as usize],
        ];
        let mut result: Vec<Vec<f32>> = vec![];
        for s in samples.clone() {
            result.push(reduction.project_vector(s));
        }
        let restored = reduction.recover_centroids(result);
        for (s, r) in zip(samples, restored) {
            let mse = zip(s, r).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / DIM as f32;
            eprintln!("dim = {DIM}, mse = {mse:.12}");
            assert!(mse <= EPSILON);
        }
    }
}

#[test]
fn relative_relation_maintenance() {
    use rand_distr::{Distribution, Normal};
    use std::iter::zip;
    const TRIES: u32 = 1000;
    const DIM: u32 = 768;
    const EPSILON: f32 = 1e-3;
    let reduction = KMeansReduction::new(DIM, 512);
    let mut rng = StdRng::from_os_rng();
    let norm_pos = Normal::new(1.0, 0.2).unwrap();
    let norm_neg = Normal::new(-1.0, 0.2).unwrap();
    let mut correct = 0;
    for _ in 0..TRIES {
        let sample_pos = (0..DIM)
            .map(|_| norm_pos.sample(&mut rng))
            .collect::<Vec<f32>>();
        let sample_neg = (0..DIM)
            .map(|_| norm_neg.sample(&mut rng))
            .collect::<Vec<f32>>();
        let proj_pos = reduction.project_vector(sample_pos);
        let proj_neg = reduction.project_vector(sample_neg);
        let centroid_pos = reduction.project_vector(vec![1.0f32; DIM as usize]);
        let centroid_neg = reduction.project_vector(vec![-1.0f32; DIM as usize]);
        let dis_pos_t = zip(&proj_pos, &centroid_pos)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            / DIM as f32;
        let dis_neg_t = zip(&proj_neg, &centroid_neg)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            / DIM as f32;
        let dis_pos_f = zip(&proj_pos, &centroid_neg)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            / DIM as f32;
        let dis_neg_f = zip(&proj_neg, &centroid_pos)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            / DIM as f32;
        if dis_pos_t <= dis_pos_f && dis_neg_t <= dis_neg_f {
            correct += 1;
        }
    }
    let rate = correct as f32 / TRIES as f32;
    eprintln!("correct = {correct}, tries = {TRIES}, rate = {rate:.3}");
    assert!(rate >= 1.0 - EPSILON);
}
