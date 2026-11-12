//! Zero-overhead parallel iteration facade with transparent sequential fallback.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! When compiled with `feature = "parallel"` on native desktop and server targets (`cfg(not(target_arch = "wasm32"))`),
//! this crate re-exports the full [Rayon](https://docs.rs/rayon) data-parallel engine (`rayon::prelude::*`).
//!
//! When compiled for WebAssembly (`cfg(target_arch = "wasm32")`) or when the `parallel` feature is disabled,
//! it seamlessly provides zero-cost inline wrappers over standard `core::iter::*` iterators.
//!
//! This enables writing performance-critical loops ONCE across the entire Spodeian ecosystem without
//! scattering repetitive `#[cfg(target_arch = "wasm32")]` blocks.
//!
//! # Examples
//!
//! ```rust
//! use spodeian_parallel::prelude::*;
//!
//! let data = vec![1, 2, 3, 4, 5];
//! let sum: i32 = data.into_par_iter().map(|x| x * 2).sum();
//! assert_eq!(sum, 30);
//! ```

#![cfg_attr(not(feature = "parallel"), no_std)]

#[cfg(not(feature = "parallel"))]
extern crate alloc;

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub use rayon::prelude::*;

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub mod prelude {
    //! Re-exports Rayon prelude on native targets with `feature = "parallel"`.
    pub use rayon::prelude::*;
}

#[cfg(any(not(feature = "parallel"), target_arch = "wasm32"))]
pub use fallback::*;

#[cfg(any(not(feature = "parallel"), target_arch = "wasm32"))]
pub mod prelude {
    //! Re-exports zero-cost sequential fallback traits on WASM or when parallel is disabled.
    pub use crate::fallback::*;
}

#[cfg(any(not(feature = "parallel"), target_arch = "wasm32"))]
mod fallback {
    /// Trait for converting a collection into a parallel iterator (sequential on WASM).
    pub trait IntoParallelIterator {
        type Item;
        type Iter: Iterator<Item = Self::Item>;

        /// Converts `self` into an iterator.
        fn into_par_iter(self) -> Self::Iter;
    }

    impl<I: IntoIterator> IntoParallelIterator for I {
        type Item = I::Item;
        type Iter = I::IntoIter;

        #[inline(always)]
        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }

    /// Trait for converting a reference to a collection into a parallel iterator (sequential on WASM).
    pub trait IntoParallelRefIterator<'data> {
        type Item;
        type Iter: Iterator<Item = Self::Item>;

        /// Creates a borrowed iterator over `self`.
        fn par_iter(&'data self) -> Self::Iter;
    }

    impl<'data, I: 'data> IntoParallelRefIterator<'data> for I
    where
        &'data I: IntoIterator,
    {
        type Item = <&'data I as IntoIterator>::Item;
        type Iter = <&'data I as IntoIterator>::IntoIter;

        #[inline(always)]
        fn par_iter(&'data self) -> Self::Iter {
            self.into_iter()
        }
    }

    /// Trait for converting a mutable reference to a collection into a parallel iterator (sequential on WASM).
    pub trait IntoParallelRefMutIterator<'data> {
        type Item;
        type Iter: Iterator<Item = Self::Item>;

        /// Creates a mutable borrowed iterator over `self`.
        fn par_iter_mut(&'data mut self) -> Self::Iter;
    }

    impl<'data, I: 'data> IntoParallelRefMutIterator<'data> for I
    where
        &'data mut I: IntoIterator,
    {
        type Item = <&'data mut I as IntoIterator>::Item;
        type Iter = <&'data mut I as IntoIterator>::IntoIter;

        #[inline(always)]
        fn par_iter_mut(&'data mut self) -> Self::Iter {
            self.into_iter()
        }
    }

    /// Parallel slice operations (sequential on WASM).
    pub trait ParallelSlice<T> {
        /// Returns parallel chunks of `chunk_size` elements (sequential on WASM).
        fn par_chunks(&self, chunk_size: usize) -> core::slice::Chunks<'_, T>;

        /// Returns parallel sliding windows of `window_size` elements (sequential on WASM).
        fn par_windows(&self, window_size: usize) -> core::slice::Windows<'_, T>;
    }

    impl<T> ParallelSlice<T> for [T] {
        #[inline(always)]
        fn par_chunks(&self, chunk_size: usize) -> core::slice::Chunks<'_, T> {
            self.chunks(chunk_size)
        }

        #[inline(always)]
        fn par_windows(&self, window_size: usize) -> core::slice::Windows<'_, T> {
            self.windows(window_size)
        }
    }

    /// Mutable parallel slice operations (sequential on WASM).
    pub trait ParallelSliceMut<T> {
        /// Returns parallel mutable chunks of `chunk_size` elements (sequential on WASM).
        fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T>;

        /// Sorts the slice in-place (unstable, sequential on WASM).
        fn par_sort_unstable(&mut self)
        where
            T: Ord;

        /// Sorts the slice in-place with a comparator (unstable, sequential on WASM).
        fn par_sort_unstable_by<F>(&mut self, compare: F)
        where
            F: FnMut(&T, &T) -> core::cmp::Ordering;

        /// Sorts the slice in-place with a key extraction function (unstable, sequential on WASM).
        fn par_sort_unstable_by_key<K, F>(&mut self, f: F)
        where
            K: Ord,
            F: FnMut(&T) -> K;

        /// Sorts the slice in-place (stable, sequential on WASM).
        fn par_sort(&mut self)
        where
            T: Ord;

        /// Sorts the slice in-place with a comparator (stable, sequential on WASM).
        fn par_sort_by<F>(&mut self, compare: F)
        where
            F: FnMut(&T, &T) -> core::cmp::Ordering;
    }

    impl<T> ParallelSliceMut<T> for [T] {
        #[inline(always)]
        fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T> {
            self.chunks_mut(chunk_size)
        }

        #[inline(always)]
        fn par_sort_unstable(&mut self)
        where
            T: Ord,
        {
            self.sort_unstable();
        }

        #[inline(always)]
        fn par_sort_unstable_by<F>(&mut self, compare: F)
        where
            F: FnMut(&T, &T) -> core::cmp::Ordering,
        {
            self.sort_unstable_by(compare);
        }

        #[inline(always)]
        fn par_sort_unstable_by_key<K, F>(&mut self, f: F)
        where
            K: Ord,
            F: FnMut(&T) -> K,
        {
            self.sort_unstable_by_key(f);
        }

        #[inline(always)]
        fn par_sort(&mut self)
        where
            T: Ord,
        {
            #[cfg(feature = "parallel")]
            self.sort();
            #[cfg(not(feature = "parallel"))]
            self.sort_unstable();
        }

        #[inline(always)]
        fn par_sort_by<F>(&mut self, compare: F)
        where
            F: FnMut(&T, &T) -> core::cmp::Ordering,
        {
            #[cfg(feature = "parallel")]
            self.sort_by(compare);
            #[cfg(not(feature = "parallel"))]
            self.sort_unstable_by(compare);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[cfg(not(feature = "parallel"))]
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_into_par_iter() {
        let v = vec![1, 2, 3, 4, 5];
        let doubled: Vec<i32> = v.into_par_iter().map(|x| x * 2).collect();
        assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_par_iter() {
        let v = vec![10, 20, 30];
        let sum: i32 = v.par_iter().sum();
        assert_eq!(sum, 60);
    }

    #[test]
    fn test_par_iter_mut() {
        let mut v = vec![1, 2, 3];
        v.par_iter_mut().for_each(|x| *x += 10);
        assert_eq!(v, vec![11, 12, 13]);
    }

    #[test]
    fn test_par_chunks() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sums: Vec<f32> = data.par_chunks(2).map(|chunk| chunk.iter().sum()).collect();
        assert_eq!(sums, vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn test_par_chunks_mut() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        data.par_chunks_mut(2).for_each(|chunk| {
            for val in chunk.iter_mut() {
                *val *= 2.0;
            }
        });
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_par_sort_unstable() {
        let mut data = vec![5, 1, 4, 2, 3];
        data.par_sort_unstable();
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }
}
