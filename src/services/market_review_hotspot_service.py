# -*- coding: utf-8 -*-
"""Market review hotspot appendix builder."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from data_provider import DataFetcherManager
from src.report_language import normalize_report_language

logger = logging.getLogger(__name__)


class MarketReviewHotspotService:
    """Build optional hotspot sections for market review push content."""

    def __init__(
        self,
        manager: Optional[DataFetcherManager] = None,
        top_n: int = 5,
    ) -> None:
        self.manager = manager or DataFetcherManager()
        self.top_n = top_n

    def build_markdown(self, region: str, language: str) -> str:
        if region != "cn":
            return ""

        review_language = normalize_report_language(language)
        sections: List[str] = []

        sectors = self._safe_get_top_sectors()
        if sectors:
            sections.append(self._format_sector_section(sectors, review_language))

        stocks = self._safe_get_hot_stocks()
        if stocks:
            sections.append(self._format_stock_section(stocks, review_language))

        return "\n\n".join(section for section in sections if section)

    def _safe_get_top_sectors(self) -> List[Dict[str, Any]]:
        try:
            top_sectors, _ = self.manager.get_sector_rankings(n=self.top_n)
            return top_sectors[: self.top_n]
        except Exception as exc:
            logger.warning("构建大盘复盘热门板块失败: %s", exc)
            return []

    def _safe_get_hot_stocks(self) -> List[Dict[str, Any]]:
        try:
            return self.manager.get_hot_stocks(n=self.top_n)[: self.top_n]
        except Exception as exc:
            logger.warning("构建大盘复盘热门股票失败: %s", exc)
            return []

    def _format_sector_section(
        self,
        sectors: List[Dict[str, Any]],
        language: str,
    ) -> str:
        title = "### Hot Sectors" if language == "en" else "### 热门板块"
        note = (
            f"> Method: ranked by daily change percentage of A-share industry sectors Top {len(sectors)}"
            if language == "en"
            else f"> 统计口径：按 A 股行业板块当日涨跌幅 Top {len(sectors)} 排序"
        )
        lines = [
            f"- **{item.get('name', '-')}**: {self._format_change(item.get('change_pct'))}"
            for item in sectors
        ]
        return "\n".join([title, note, *lines])

    def _format_stock_section(
        self,
        stocks: List[Dict[str, Any]],
        language: str,
    ) -> str:
        title = "### Hot Stocks" if language == "en" else "### 热门股票"
        note = (
            f"> Method: ranked by daily change percentage first and turnover second among A-share stocks Top {len(stocks)} (ETF excluded)"
            if language == "en"
            else f"> 统计口径：按 A 股个股当日涨跌幅优先、成交额次排序，取 Top {len(stocks)}（已剔除 ETF）"
        )
        lines = [self._format_stock_line(item, language) for item in stocks]
        return "\n".join([title, note, *lines])

    def _format_stock_line(self, stock: Dict[str, Any], language: str) -> str:
        code = str(stock.get("code", "")).strip()
        name = str(stock.get("name", "")).strip() or code or "-"
        amount_text = self._format_amount(stock.get("amount"), language)
        turnover_rate = self._safe_float(stock.get("turnover_rate"))

        parts = [
            f"{self._format_change(stock.get('change_pct'))}",
            f"turnover {amount_text}" if language == "en" else f"成交额 {amount_text}",
        ]
        if turnover_rate is not None:
            parts.append(
                f"turnover rate {turnover_rate:.1f}%"
                if language == "en"
                else f"换手率 {turnover_rate:.1f}%"
            )
        return f"- **{name} ({code})**: {', '.join(parts)}"

    @staticmethod
    def _format_change(value: Any) -> str:
        numeric_value = MarketReviewHotspotService._safe_float(value)
        if numeric_value is None:
            return "N/A"
        return f"{numeric_value:+.2f}%"

    @staticmethod
    def _format_amount(value: Any, language: str) -> str:
        amount = MarketReviewHotspotService._safe_float(value)
        if amount is None:
            return "N/A"

        if language == "en":
            if abs(amount) >= 1e9:
                return f"CNY {amount / 1e9:.1f}B"
            if abs(amount) >= 1e6:
                return f"CNY {amount / 1e6:.1f}M"
            return f"CNY {amount:.0f}"

        if abs(amount) >= 1e8:
            return f"{amount / 1e8:.1f}亿"
        if abs(amount) >= 1e4:
            return f"{amount / 1e4:.0f}万"
        return f"{amount:.0f}元"

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric_value):
            return None
        return numeric_value
