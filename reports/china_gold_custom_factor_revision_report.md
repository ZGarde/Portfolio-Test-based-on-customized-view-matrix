# China Gold Custom Factor Revision Report

## 1. Why The Old Signal Was Not The Intended Signal

The previous China gold custom factor mainly used `china_gold_evening_ret`, which is the percentage change from one evening price to the next evening price. That measures evening-to-evening daily momentum in China gold. It does not isolate the information shock that occurred inside the current China trading day.

## 2. Why The New Signal Matches The Economic Hypothesis

The revised primary signal is:

```text
china_gold_morning_to_evening_ret = china_gold_evening / china_gold_morning - 1
```

This measures the directional move from China morning/open to China evening in the same trading day. A positive value means China gold strengthened during that China trading window, which better represents the intended bullish information shock. The Black-Litterman custom view uses the rolling z-score version, `china_gold_morning_to_evening_z`.

## 3. Why There Is No Mechanical Shift(1)

The hypothesis assumes China evening and US gold reaction are separated by only a few hours. Therefore, the primary test should prefer same-day alignment instead of mechanically shifting the signal to the next day. The next-observed US gold return is still reported as a comparison, but it is not treated as the only valid target.

## 4. Current Limitation

The project does not currently have true US gold open/reopen intraday prices. The tests therefore use daily US gold close-to-close returns as proxy targets. This can test whether the China morning-to-evening shock relates to same-day or next-observed US gold daily returns, but it cannot strictly prove US high-open behavior.

## 5. How The Factor Enters Black-Litterman

The factor does not directly set Gold weight. It is mapped into a Black-Litterman absolute view:

```text
P = [1, 0, 0, 0, 0]
Q = dynamic annualized expected return impact
confidence = dynamic confidence
```

For the latest available `china_gold_morning_to_evening_z`:

```text
if z is NaN: skip the view
if abs(z) < 1.0: Q = 0, confidence = 0.01, not a strong active view
if abs(z) >= 1.0:
    Q = clip(0.01 * z, -0.03, 0.03)
    confidence = min(0.10 + 0.10 * max(abs(z) - 1.0, 0), 0.30)
```

The Black-Litterman posterior expected returns and the optimizer then determine the final Gold weight together with covariance, constraints, and other active views.

## Classification

This revised China gold factor remains a research-only proxy. It is not a confirmed signal until the China data source is confirmed as the intended tradable futures series and US gold intraday open/reopen prices are available for strict timing validation.
