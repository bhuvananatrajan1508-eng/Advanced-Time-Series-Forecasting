**RESULTS & DISCUSSION**

**1. Results**

The proposed attention-based time series forecasting model was trained on multivariate climate data, including temperature, pressure, humidity, wind speed, and wind direction. After training, the model successfully learned temporal dependencies within the historical time window and produced accurate future predictions.

The attention mechanism provided an additional layer of interpretability by assigning importance weights to historical time steps. Visualization of attention weights revealed that the model consistently focused more on recent observations, while still retaining selective attention to certain older time steps. This confirms that both short-term and contextual long-term patterns influence forecasting performance.
Heatmap analysis showed a non-uniform distribution of attention, indicating that the model does not treat all historical inputs equally. Instead, it selectively emphasizes informative time steps that contribute most to prediction accuracy.

Overlaying attention weights on the temperature time series further demonstrated that attention peaks often coincide with rapid changes or significant fluctuations in temperature, suggesting that the model prioritizes critical transition periods rather than stable intervals.

**2. Discussion**

The results highlight the effectiveness of attention mechanisms in enhancing both model performance and interpretability. Traditional deep learning models such as LSTM or GRU act as black boxes, whereas attention enables insight into the model’s decision-making process.

The dominance of recent time steps in attention distribution aligns with the nature of climate data, where recent weather conditions strongly influence near-future outcomes. However, the presence of non-zero attention for earlier time steps indicates that the model captures delayed or cumulative effects, such as pressure trends or humidity buildup.

Multivariate attention analysis revealed that temperature and wind-related features had higher cumulative attention scores compared to other variables. This suggests that these features play a more significant role in short-term forecasting within the selected dataset.
Overall, the attention-based approach improves trustworthiness and usability of the forecasting model, making it suitable for real-world decision-support systems where explainability is crucial.

**MATHEMATICAL EXPLANATION OF ATTENTION**

Let the input sequence be:

𝑋
=
{
𝑥
1
,
𝑥
2
,
…
,
𝑥
𝑇
}
X={x
1
	​

,x
2
	​

,…,x
T
	​

}

where:

𝑇
T = number of historical time steps

𝑥
𝑡
∈
𝑅
𝑑
x
t
	​

∈R
d
 represents a multivariate feature vector at time 
𝑡
t

**Step 1: Score Calculation**

Each hidden state 
ℎ
𝑡
h
t
	​

 is mapped to an attention score:

𝑒
𝑡
=
𝑓
(
ℎ
𝑡
)
e
t
	​

=f(h
t
	​

)

where 
𝑓
(
⋅
)
f(⋅) is a learnable function (e.g., linear layer or MLP).

**Step 2: Attention Weight (Softmax)**

The raw scores are normalized using Softmax:

𝛼
𝑡
=
exp
⁡
(
𝑒
𝑡
)
∑
𝑘
=
1
𝑇
exp
⁡
(
𝑒
𝑘
)
α
t
	​

=
∑
k=1
T
	​

exp(e
k
	​

)
exp(e
t
	​

)
	​


where:

𝛼
𝑡
α
t
	​

 = attention weight for time step 
𝑡
t

∑
𝑡
=
1
𝑇
𝛼
𝑡
=
1
∑
t=1
T
	​

α
t
	​

=1

**Step 3: Context Vector**

The final context vector is computed as:

𝑐
=
∑
𝑡
=
1
𝑇
𝛼
𝑡
ℎ
𝑡
c=
t=1
∑
T
	​

α
t
	​

h
t
	​


This context vector represents a weighted summary of historical information and is used for final prediction.

**MULTIVARIATE ATTENTION ANALYSIS**

**1. Feature-wise Contribution**

In multivariate time series, each input vector contains multiple features:

𝑥
𝑡
=
[
𝑥
𝑡
(
1
)
,
𝑥
𝑡
(
2
)
,
…
,
𝑥
𝑡
(
𝑑
)
]
x
t
	​

=[x
t
(1)
	​

,x
t
(2)
	​

,…,x
t
(d)
	​

]

While temporal attention focuses on when to attend, multivariate analysis helps understand what features matter most.

**3. Observations**

Temperature (T (degC)) exhibited the highest importance score, confirming its dominant influence on forecasting.

Wind speed and direction showed moderate importance, indicating their role during sudden climate changes.

Humidity and pressure contributed more steadily across time, suggesting long-term contextual influence rather than immediate impact.

**4. Discussion of Multivariate Attention**

The multivariate attention analysis demonstrates that the model effectively integrates heterogeneous climate signals rather than relying on a single dominant variable. This balanced utilization of features improves robustness and generalization.

Such analysis is particularly valuable for climate and environmental forecasting tasks, where inter-feature dependencies play a critical role.

**FINAL SUMMARY (FOR CONCLUSION SLIDE)**

1. Attention improves forecast accuracy and interpretability

2. Recent time steps are most influential, but long-term context matters

3. Temperature and wind features dominate short-term predictions

4. Attention enables explainable AI for time-series forecasting
