# regularization-explorer
### Model Interpretation & Analysis

The generated regularization paths illustrate the fundamental difference between L1 (Lasso) and L2 (Ridge) regularization behaviors:

* **Sparsity with L1 (Lasso):** As observed in the left plot, L1 regularization effectively forces coefficients to exactly zero, especially at low values of C (strong regularization). Features like `senior_citizen`, `has_dependents`, and `has_partner` are "zeroed out" first, remaining at zero until C increases significantly. This demonstrates L1's role in **automated feature selection**, effectively identifying the most influential predictors by eliminating the noise.
* **Shrinkage with L2 (Ridge):** The right plot shows that L2 regularization shrinks coefficients toward zero but rarely eliminates them entirely. Even at very low C values, most features retain non-zero coefficient values. This indicates L2's tendency to **preserve feature information** and handle multicollinearity by distributing the weight across all predictors.
* **Key Predictors:** In both regularization paths, `tenure` (orange) and `monthly_charges` (green) emerge as the most robust and influential predictors, maintaining significant magnitude even under strong regularization, whereas other features demonstrate higher sensitivity.

**Recommendation:** Given the need for a clear and interpretable model in churn prediction, L1 is preferable if the business requires a simplified model with fewer features. However, if predictive accuracy is paramount and we wish to retain the full context of all customer attributes, L2 provides a more stable and balanced model.