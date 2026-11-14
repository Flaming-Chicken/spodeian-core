//! Exact multi-objective Pareto dominance filtering, 2D hypervolume, and Utopia knee-point selection.

use serde::{Deserialize, Serialize};

/// Optimization direction for a single objective axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

/// Returns `true` if `a` strictly Pareto-dominates `b` under `directions`
/// (as good or better on all objectives, strictly better on at least one).
pub fn dominates(a: &[f64], b: &[f64], directions: &[ObjectiveDirection]) -> bool {
    if a.len() != directions.len() || b.len() != directions.len() {
        return false;
    }
    let mut strictly_better = false;
    for ((&va, &vb), &dir) in a.iter().zip(b).zip(directions) {
        if !va.is_finite() || !vb.is_finite() {
            return false;
        }
        match dir {
            ObjectiveDirection::Minimize => {
                if va > vb + 1e-12 {
                    return false;
                }
                if va < vb - 1e-12 {
                    strictly_better = true;
                }
            }
            ObjectiveDirection::Maximize => {
                if va < vb - 1e-12 {
                    return false;
                }
                if va > vb + 1e-12 {
                    strictly_better = true;
                }
            }
        }
    }
    strictly_better
}

/// Exact $O(N^2 d)$ non-dominated index filter returning indices of points not dominated by any other point.
pub fn exact_non_dominated_indices(
    points: &[Vec<f64>],
    directions: &[ObjectiveDirection],
) -> Vec<usize> {
    let mut frontier = Vec::new();
    for (i, pi) in points.iter().enumerate() {
        if pi.len() != directions.len() || pi.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let is_dominated = points
            .iter()
            .enumerate()
            .any(|(j, pj)| i != j && dominates(pj, pi, directions));
        if !is_dominated {
            frontier.push(i);
        }
    }
    frontier
}

/// Verifies that no point in `points` dominates another point in `points`.
pub fn is_strict_pareto_frontier(
    points: &[Vec<f64>],
    directions: &[ObjectiveDirection],
) -> bool {
    for (i, pi) in points.iter().enumerate() {
        if pi.len() != directions.len() || pi.iter().any(|v| !v.is_finite()) {
            return false;
        }
        for (j, pj) in points.iter().enumerate() {
            if i != j && dominates(pj, pi, directions) {
                return false;
            }
        }
    }
    true
}

/// Selects the Utopia-normalized knee point index (minimum Euclidean distance to the normalized ideal point).
pub fn select_utopia_knee_index(
    points: &[Vec<f64>],
    directions: &[ObjectiveDirection],
) -> Option<usize> {
    if points.is_empty() || directions.is_empty() {
        return None;
    }
    let d = directions.len();
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for p in points {
        if p.len() != d {
            return None;
        }
        for k in 0..d {
            mins[k] = mins[k].min(p[k]);
            maxs[k] = maxs[k].max(p[k]);
        }
    }

    let mut best_idx = None;
    let mut best_dist = f64::INFINITY;
    for (i, p) in points.iter().enumerate() {
        let mut dist_sq = 0.0;
        for (k, &dir) in directions.iter().enumerate() {
            let span = (maxs[k] - mins[k]).max(1e-12);
            let norm = match dir {
                ObjectiveDirection::Minimize => (p[k] - mins[k]) / span,
                ObjectiveDirection::Maximize => (maxs[k] - p[k]) / span,
            };
            dist_sq += norm * norm;
        }
        if dist_sq < best_dist {
            best_dist = dist_sq;
            best_idx = Some(i);
        }
    }
    best_idx
}

/// Computes exact 2D dominated hypervolume for minimization points `(x, y)` bounded above by `ref_point`.
pub fn hypervolume_2d(points_min_min: &[(f64, f64)], ref_point: (f64, f64)) -> f64 {
    let mut pts: Vec<(f64, f64)> = points_min_min
        .iter()
        .copied()
        .filter(|&(x, y)| x < ref_point.0 && y < ref_point.1 && x.is_finite() && y.is_finite())
        .collect();
    if pts.is_empty() {
        return 0.0;
    }
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut hv = 0.0;
    let mut best_y = ref_point.1;
    for i in 0..pts.len() {
        let (x, y) = pts[i];
        best_y = best_y.min(y);
        let next_x = if i + 1 < pts.len() {
            pts[i + 1].0.min(ref_point.0)
        } else {
            ref_point.0
        };
        hv += (next_x - x).max(0.0) * (ref_point.1 - best_y).max(0.0);
    }
    hv
}
