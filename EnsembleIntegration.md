# Ensemble Integration and Anomaly Agent Explanation

## Anomaly Agent Explanation
- The `AnomalyAgent` currently uses an autoencoder stub: it treats each input window `(n_assets, lookback, n_features)`
  as a flattened vector and minimizes mean squared reconstruction error.
- **Large loss magnitude** is expected because:
  - Raw price columns (Open/High/Low/Close) remain unscaled, often in tens or hundreds, driving up MSE.
  - The window flatten dimension is high (e.g. 50 days × 21 features = 1050 dims), so total MSE accumulates across dimensions.
  - To obtain meaningful anomaly scores:
    1. Normalize all features to a common scale (0–1) before training the autoencoder.
    2. Use per-asset or per-feature relative errors (e.g. percentage error) instead of raw squared error.
    3. Compute anomaly score per asset as the mean reconstruction error over its window.
    4. Train a lightweight bottleneck network to avoid overfitting and capture core patterns.

## Ensemble Integration Workflow
1. **Pre-train sub-agents** (autoencoder, short-term predictor, etc.) and save their weights:
   - `models/anomaly.pt`, `models/short_term.pt`, `models/long_term.pt`, `models/sentiment.pt`
2. **Extend `EnsembleAgent` to load sub-agent weights** in its constructor:
   ```python
   class EnsembleAgent(BaseAgent):
       def __init__(...):
           # Load pre-trained sub-agents
           self.anomaly = AnomalyAgent()
           self.anomaly.load_models('models/anomaly.pt')
           self.short_pred = ShortTermPredictorAgent()
           self.short_pred.load_models('models/short_term.pt')
           # ... similarly for long_term, sentiment
   ```
3. **Modify `choose_action()`** to incorporate sub-agent outputs:
   ```python
   def choose_action(self, observation):
       raw_flat = observation.flatten()
       a_score = self.anomaly.predict(observation)
       st_pred = self.short_pred.predict(observation)
       lt_pred = self.long_pred.predict(observation)
       sent_bias = self.sentiment.predict(observation)
       sub_flat = np.concatenate([a_score, st_pred, lt_pred, sent_bias])
       total_feat = np.concatenate([raw_flat, sub_flat])
       x = torch.tensor(total_feat).unsqueeze(0)
       dist, value = self.aggregator(x)
       ...
   ```
4. **Train aggregator via PPO** with frozen sub-agents:
   ```bash
   python main.py --config config_ensemble.yaml --mode train
   ```
   where `config_ensemble.yaml` sets `agent.type: ensemble` and appropriate checkpoint files.
5. **Optional Joint Fine-Tuning**
   - Unfreeze sub-agents in `EnsembleAgent.learn()`
   - Add auxiliary MSE losses for each `predict()`
   - Optimize combined PPO + λ * sum(aux_losses)

This approach ensures each specialized module captures its niche (anomalies, short-term, etc.),
while the aggregator learns to weight and integrate those signals into end-to-end portfolio decisions.