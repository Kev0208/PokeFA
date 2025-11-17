from __future__ import annotations
import time
from typing import Dict, Iterable, Optional
from serpapi import GoogleSearch

class SerpApiImages:
    def __init__(self, api_key: str, rps: float = 1.5, hl: str = "en", gl: str = "us"):
        if not api_key:
            raise RuntimeError("api_key required")
        self.api_key = api_key
        self.delay = 1.0 / max(rps, 0.1)
        self.hl, self.gl = hl, gl

    def _sleep(self) -> None:
        time.sleep(self.delay)

    def search_images(self, query: str, page: int = 0, safe: str = "active") -> Dict:
        params = {
            "engine": "google_images",
            "q": query,
            "ijn": page,
            "api_key": self.api_key,
            "safe": safe,
            "hl": self.hl,
            "gl": self.gl,
        }
        self._sleep()
        return GoogleSearch(params).get_dict()

    def iter_image_results(self, query: str, max_pages: int = 50, safe: str = "active") -> Iterable[Dict]:
        page = 0
        while page < max_pages:
            data = self.search_images(query=query, page=page, safe=safe)
            items = data.get("images_results", []) or []
            if not items:
                break
            for it in items:
                yield it
            page += 1
