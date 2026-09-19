"""
Vectorized Fuzzy Inference System (FIS) for Multi-Model Fusion
Pure Python / NumPy implementation (zero external C dependencies, 100% Colab compatible).
Supports both Mamdani (Centroid Defuzzification) and Takagi-Sugeno inference.
"""

import numpy as np
from typing import Union, Tuple, List, Dict

class FuzzyMembership:
    """Membership function definitions for linguistic terms Low, Medium, and High."""
    
    @staticmethod
    def trimf(x: np.ndarray, params: Tuple[float, float, float]) -> np.ndarray:
        """Triangular membership function."""
        a, b, c = params
        return np.maximum(0.0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

    @staticmethod
    def trapmf(x: np.ndarray, params: Tuple[float, float, float, float]) -> np.ndarray:
        """Trapezoidal membership function."""
        a, b, c, d = params
        term1 = (x - a) / (b - a + 1e-9)
        term2 = (d - x) / (d - c + 1e-9)
        return np.maximum(0.0, np.minimum(np.minimum(term1, 1.0), term2))

    @staticmethod
    def low_half_trap(x: np.ndarray, a: float = 0.15, b: float = 0.40) -> np.ndarray:
        """Left-shoulder trapezoidal membership for 'Low'."""
        res = np.ones_like(x, dtype=np.float64)
        mask_decay = (x > a) & (x < b)
        res[mask_decay] = (b - x[mask_decay]) / (b - a + 1e-9)
        res[x >= b] = 0.0
        return np.clip(res, 0.0, 1.0)

    @staticmethod
    def medium_tri(x: np.ndarray, a: float = 0.25, b: float = 0.50, c: float = 0.75) -> np.ndarray:
        """Symmetric triangular membership for 'Medium'."""
        return np.clip(FuzzyMembership.trimf(x, (a, b, c)), 0.0, 1.0)

    @staticmethod
    def high_half_trap(x: np.ndarray, a: float = 0.60, b: float = 0.85) -> np.ndarray:
        """Right-shoulder trapezoidal membership for 'High'."""
        res = np.zeros_like(x, dtype=np.float64)
        mask_rise = (x > a) & (x < b)
        res[mask_rise] = (x[mask_rise] - a) / (b - a + 1e-9)
        res[x >= b] = 1.0
        return np.clip(res, 0.0, 1.0)


class FuzzyFusionEngine:
    """
    9-Rule Mamdani & Sugeno Fuzzy Inference Engine for fusing ResNet-101 and DenseNet-201.
    """
    def __init__(
        self,
        low_bounds: Tuple[float, float] = (0.15, 0.40),
        med_bounds: Tuple[float, float, float] = (0.25, 0.50, 0.75),
        high_bounds: Tuple[float, float] = (0.60, 0.85),
        n_disc: int = 101,
        mode: str = "mamdani"
    ):
        self.low_bounds = low_bounds
        self.med_bounds = med_bounds
        self.high_bounds = high_bounds
        self.n_disc = n_disc
        self.mode = mode.lower()
        
        # Discretized output universe Y in [0, 1] for Mamdani defuzzification
        self.y_universe = np.linspace(0.0, 1.0, self.n_disc)
        
        # Precompute output membership sets on Y universe
        self.out_low = FuzzyMembership.low_half_trap(self.y_universe, low_bounds[0], low_bounds[1])
        self.out_med = FuzzyMembership.medium_tri(self.y_universe, med_bounds[0], med_bounds[1], med_bounds[2])
        self.out_high = FuzzyMembership.high_half_trap(self.y_universe, high_bounds[0], high_bounds[1])

        # Sugeno singleton constants for Low, Medium, High
        self.sugeno_singletons = {
            "LOW": 0.15,
            "MED": 0.50,
            "HIGH": 0.85
        }

    def fuzzify(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert scalar probability array into membership degrees [mu_L, mu_M, mu_H]."""
        p = np.asarray(p, dtype=np.float64)
        mu_l = FuzzyMembership.low_half_trap(p, self.low_bounds[0], self.low_bounds[1])
        mu_m = FuzzyMembership.medium_tri(p, self.med_bounds[0], self.med_bounds[1], self.med_bounds[2])
        mu_h = FuzzyMembership.high_half_trap(p, self.high_bounds[0], self.high_bounds[1])
        return mu_l, mu_m, mu_h

    def infer_mamdani(self, p_resnet: np.ndarray, p_densenet: np.ndarray) -> np.ndarray:
        """
        Execute 9 Mamdani rules and Centroid Defuzzification over batch of probabilities.
        """
        p_res = np.atleast_1d(np.asarray(p_resnet, dtype=np.float64))
        p_den = np.atleast_1d(np.asarray(p_densenet, dtype=np.float64))
        assert p_res.shape == p_den.shape, "Prediction arrays must have matching shapes"

        N = len(p_res)
        l1, m1, h1 = self.fuzzify(p_res)
        l2, m2, h2 = self.fuzzify(p_den)

        # Compute firing strengths for the 9 rules using minimum t-norm
        # R1: IF L1 AND L2 -> LOW
        w1 = np.minimum(l1, l2)
        # R2: IF L1 AND M2 -> LOW
        w2 = np.minimum(l1, m2)
        # R3: IF L1 AND H2 -> MED
        w3 = np.minimum(l1, h2)
        # R4: IF M1 AND L2 -> LOW
        w4 = np.minimum(m1, l2)
        # R5: IF M1 AND M2 -> MED
        w5 = np.minimum(m1, m2)
        # R6: IF M1 AND H2 -> HIGH
        w6 = np.minimum(m1, h2)
        # R7: IF H1 AND L2 -> MED
        w7 = np.minimum(h1, l2)
        # R8: IF H1 AND M2 -> HIGH
        w8 = np.minimum(h1, m2)
        # R9: IF H1 AND H2 -> HIGH
        w9 = np.minimum(h1, h2)

        # Aggregate rule firings for each consequent linguistic category (MAX s-norm)
        fire_low = np.maximum.reduce([w1, w2, w4])       # Shape: (N,)
        fire_med = np.maximum.reduce([w3, w5, w7])       # Shape: (N,)
        fire_high = np.maximum.reduce([w6, w8, w9])      # Shape: (N,)

        # Defuzzification across the discretized universe
        # Outer broadcast: (N, n_disc)
        clipped_low = np.minimum(fire_low[:, None], self.out_low[None, :])
        clipped_med = np.minimum(fire_med[:, None], self.out_med[None, :])
        clipped_high = np.minimum(fire_high[:, None], self.out_high[None, :])

        # Overall aggregated membership curve: max across consequents
        agg_curve = np.maximum.reduce([clipped_low, clipped_med, clipped_high])

        # Centroid Defuzzification (Center of Gravity / Area)
        numerator = np.sum(agg_curve * self.y_universe, axis=1)
        denominator = np.sum(agg_curve, axis=1)

        # Fallback to simple average in the rare case denominator == 0
        fallback = 0.5 * (p_res + p_den)
        crisp_outputs = np.where(denominator > 1e-7, numerator / (denominator + 1e-9), fallback)

        return np.clip(crisp_outputs, 0.0, 1.0)

    def infer_sugeno(self, p_resnet: np.ndarray, p_densenet: np.ndarray) -> np.ndarray:
        """
        Execute 0-th order Takagi-Sugeno inference (Weighted Average of singletons).
        """
        p_res = np.atleast_1d(np.asarray(p_resnet, dtype=np.float64))
        p_den = np.atleast_1d(np.asarray(p_densenet, dtype=np.float64))

        l1, m1, h1 = self.fuzzify(p_res)
        l2, m2, h2 = self.fuzzify(p_den)

        # Rule weights
        w = [
            np.minimum(l1, l2),  # R1 -> LOW
            np.minimum(l1, m2),  # R2 -> LOW
            np.minimum(l1, h2),  # R3 -> MED
            np.minimum(m1, l2),  # R4 -> LOW
            np.minimum(m1, m2),  # R5 -> MED
            np.minimum(m1, h2),  # R6 -> HIGH
            np.minimum(h1, l2),  # R7 -> MED
            np.minimum(h1, m2),  # R8 -> HIGH
            np.minimum(h1, h2)   # R9 -> HIGH
        ]
        consequents = [
            self.sugeno_singletons["LOW"],
            self.sugeno_singletons["LOW"],
            self.sugeno_singletons["MED"],
            self.sugeno_singletons["LOW"],
            self.sugeno_singletons["MED"],
            self.sugeno_singletons["HIGH"],
            self.sugeno_singletons["MED"],
            self.sugeno_singletons["HIGH"],
            self.sugeno_singletons["HIGH"]
        ]

        w_sum = sum(w)
        num = sum(w_i * z_i for w_i, z_i in zip(w, consequents))
        fallback = 0.5 * (p_res + p_den)
        crisp_outputs = np.where(w_sum > 1e-7, num / (w_sum + 1e-9), fallback)
        return np.clip(crisp_outputs, 0.0, 1.0)

    def predict_proba(self, p_resnet: np.ndarray, p_densenet: np.ndarray) -> np.ndarray:
        """Unified inference caller based on initialized mode."""
        if self.mode == "sugeno":
            return self.infer_sugeno(p_resnet, p_densenet)
        return self.infer_mamdani(p_resnet, p_densenet)


def calibrate_fuzzy_threshold(
    p_resnet_val: np.ndarray,
    p_densenet_val: np.ndarray,
    y_val: np.ndarray,
    engine: FuzzyFusionEngine
) -> Tuple[float, float]:
    """
    Search for the optimal decision threshold tau in [0.30, 0.70] that maximizes F1-score on validation set.
    """
    p_fused = engine.predict_proba(p_resnet_val, p_densenet_val)
    best_thresh = 0.50
    best_f1 = 0.0

    thresholds = np.linspace(0.30, 0.70, 41)
    for t in thresholds:
        preds = (p_fused >= t).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        fn = np.sum((preds == 0) & (y_val == 1))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    return best_thresh, best_f1

if __name__ == "__main__":
    # Self-test unit verification
    engine = FuzzyFusionEngine()
    
    # Test extreme normal agreement
    p_low = engine.predict_proba([0.05], [0.08])[0]
    # Test extreme osteoporosis agreement
    p_high = engine.predict_proba([0.92], [0.89])[0]
    # Test conflict case (ResNet high, DenseNet low)
    p_conflict = engine.predict_proba([0.90], [0.15])[0]
    # Test moderate ambiguity
    p_mid = engine.predict_proba([0.52], [0.55])[0]

    print("FUZZY ENGINE UNIT VERIFICATION:")
    print(f"Agreement Low  [0.05, 0.08] -> Output: {p_low:.4f}  (Expected < 0.30)")
    print(f"Agreement High [0.92, 0.89] -> Output: {p_high:.4f} (Expected > 0.70)")
    print(f"Conflict Case  [0.90, 0.15] -> Output: {p_conflict:.4f} (Expected ~0.45 - 0.55)")
    print(f"Ambiguity Case [0.52, 0.55] -> Output: {p_mid:.4f} (Expected ~0.50)")
    assert p_low < 0.30, "Low agreement test failed"
    assert p_high > 0.70, "High agreement test failed"
    print("All Fuzzy Engine assertions passed successfully!")
