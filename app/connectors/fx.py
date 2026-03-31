from datetime import date

from app.connectors.base import BaseConnector, Document

MOCK_RATES_USD: dict[str, float] = {
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 151.23,
    "CNY": 7.25,
    "HKD": 7.82,
    "SGD": 1.35,
    "AUD": 1.53,
    "CAD": 1.37,
    "CHF": 0.90,
    "INR": 83.50,
}


class FXConnector(BaseConnector):
    def fetch(
        self,
        base_currency: str = "USD",
        date_str: str | None = None,
        **kwargs,
    ) -> list[Document]:
        if date_str is None:
            date_str = date.today().isoformat()
        lines = [
            f"1 {base_currency} = {rate} {target} as of {date_str}"
            for target, rate in MOCK_RATES_USD.items()
        ]
        content = "\n".join(lines)
        return [
            Document(
                id=self.make_id("fx", f"{base_currency}_{date_str}"),
                content=content,
                source_type="fx",
                title=f"FX Rates ({base_currency} base, {date_str})",
                metadata={
                    "base_currency": base_currency,
                    "date": date_str,
                    "source": "mock",
                    "rates_count": len(MOCK_RATES_USD),
                },
            )
        ]
