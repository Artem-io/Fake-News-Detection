import csv
from pathlib import Path


CSV_PATH = Path(__file__).parent / "news_media_reliability.csv"


class SourceVerifier:
    def __init__(self):
        # domain -> reliability_label (1=reliable, 0=mixed, -1=unreliable)
        self.domains: dict[str, int] = {}
        self._load_csv()

    def _load_csv(self):
        if not CSV_PATH.exists():
            print(f"Warning: {CSV_PATH} not found. Source verification disabled.")
            return

        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain = row["domain"].lower().strip()
                label = int(row["reliability_label"])
                self.domains[domain] = label

        print(f"Loaded {len(self.domains)} domains from {CSV_PATH.name}")

    def _normalize_domain(self, url: str) -> str:
        url = url.lower().strip()
        if url.startswith("http://"):
            url = url[7:]
        elif url.startswith("https://"):
            url = url[8:]
        if url.startswith("www."):
            url = url[4:]
        if "/" in url:
            url = url.split("/")[0]
        return url

    def _lookup(self, domain: str) -> tuple[str, int | None]:
        # Try exact match then remove subdomains
        parts = domain.split(".")
        for i in range(len(parts) - 1):
            candidate = ".".join(parts[i:])
            if candidate in self.domains:
                return candidate, self.domains[candidate]
        return domain, None

    def check(self, url: str) -> dict:
        domain = self._normalize_domain(url)
        matched, label = self._lookup(domain)

        if label is None:
            return {"domain": domain, "status": "NOT_FOUND", "label": None}

        status_map = {1: "RELIABLE", 0: "MIXED", -1: "UNRELIABLE"}
        return {"domain": matched, "status": status_map[label], "label": label}
