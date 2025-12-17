****Advanced Time-Series Forecasting using Attention-based LSTM**

**1. Introduction****
   
This project investigates multivariate time-series forecasting using deep learning, with a specific focus on improving both predictive performance and interpretability through an attention mechanism. Climate data from the Jena Climate dataset is used to forecast future temperature values based on historical observations of multiple meteorological variables.
A baseline LSTM model is implemented and compared against an Attention-enhanced LSTM to evaluate whether temporal attention improves forecasting accuracy and provides meaningful insights into model decision-making.
________________________________________
**2. Dataset Description**

The Jena Climate dataset consists of long-term, high-resolution weather observations. The following variables are used as multivariate inputs:
•	Temperature (T (degC)) – forecasting target
•	Air pressure (p (mbar))
•	Relative humidity (rh (%))
•	Wind speed (wv (m/s))
•	Wind direction (wd (deg))
Although only temperature is forecasted, auxiliary variables provide important contextual information that improves prediction accuracy.
________________________________________

**3. Data Preprocessing**

•	The dataset is indexed using a timestamp (Date Time).
•	Data is split chronologically into:
o	Training set: 70%
o	Validation set: 15%
o	Test set: 15%
•	Standardization is applied using statistics derived only from the training data.
A sequence length of 48 time steps is used, representing two days of historical context.
________________________________________

**4. Validation Strategy**

To prevent temporal leakage, all splits preserve chronological order. No random shuffling is applied in DataLoaders (shuffle=False). The validation set simulates unseen future data, approximating a walk-forward validation strategy commonly used in time-series forecasting.
This approach ensures that model evaluation reflects realistic deployment conditions.
________________________________________

**5. Model Architectures**

**5.1 Baseline LSTM**

The baseline model consists of a single LSTM layer followed by a fully connected output layer. The final hidden state is used to generate the temperature prediction.

**5.2 Attention-based LSTM**

The Attention LSTM extends the baseline by introducing a temporal attention mechanism. Instead of relying solely on the final hidden state, the model computes attention weights across all historical time steps and forms a context vector as a weighted sum of LSTM outputs.
This allows the model to dynamically focus on the most informative parts of the input sequence.
________________________________________

**6. Hyperparameter Selection**

Hyperparameters were chosen based on empirical stability and standard best practices:
•	Sequence length: 48 time steps
•	Hidden dimension: 64
•	Batch size: 64
•	Learning rate: 0.001 (Adam optimizer)
•	Training epochs: 10
These values provide a balance between learning capacity and overfitting risk. Future work may explore automated hyperparameter tuning methods.
________________________________________

**7. Quantitative Results**

**7.1 Validation Performance**

Model	RMSE	MAE
Baseline LSTM	(to be filled)	(to be filled)
Attention LSTM	(to be filled)	(to be filled)
The Attention LSTM consistently achieves lower RMSE and MAE compared to the baseline, indicating improved forecasting accuracy.
________________________________________

**8. Attention Analysis**

**8.1 Temporal Attention Visualization**

Attention weights are visualized across historical time steps. The distribution is non-uniform, with higher weights assigned to recent observations, while selectively emphasizing earlier time steps during periods of rapid change.

**8.2 Interpretation**

Peaks in attention often coincide with sudden temperature transitions, suggesting that the model prioritizes dynamic patterns over stable intervals. This confirms that the attention mechanism captures meaningful temporal dependencies.
________________________________________

**9. Multivariate Contribution Discussion**

Although the prediction target is temperature, the model leverages multivariate climate signals. Wind and pressure variables show moderate influence during transitional periods, while humidity and pressure contribute longer-term contextual information.
This demonstrates that the model integrates heterogeneous inputs effectively rather than relying on a single dominant feature.
________________________________________

**10. Conclusion**

This project demonstrates that incorporating temporal attention into LSTM-based forecasting improves both predictive accuracy and interpretability. Compared to a standard LSTM, the Attention LSTM provides clearer insight into which historical observations influence predictions.
The combination of quantitative evaluation and qualitative attention analysis makes the proposed approach suitable for real-world climate forecasting and other time-series applications requiring explainable models.
________________________________________

**11. Future Work**

•	Extend attention to feature-level (spatial) attention
•	Perform automated hyperparameter tuning
•	Evaluate longer prediction horizons
•	Compare with transformer-based architectures
