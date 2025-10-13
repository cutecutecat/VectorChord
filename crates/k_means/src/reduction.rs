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
        Self {
            projector,
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
}
