from __future__ import annotations


class PriceTable:
    def __init__(self, prices: dict[str, tuple[float, float]]) -> None:
        self.prices = prices

    def require(self, models: tuple[str, ...]) -> None:
        missing = [m for m in models if m not in self.prices]
        if missing:
            raise ValueError(f"No price configured for: {', '.join(missing)} (set PRICING_JSON)")

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price_in, price_out = self.prices[model]
        return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 8)
