# -*- coding: utf-8 -*-
"""Tests for market review hotspot appendix builder."""

import unittest

from src.services.market_review_hotspot_service import MarketReviewHotspotService


class _StubManager:
    def __init__(self, sectors=None, stocks=None, sector_error=None, stock_error=None):
        self.sectors = sectors if sectors is not None else ([], [])
        self.stocks = stocks if stocks is not None else []
        self.sector_error = sector_error
        self.stock_error = stock_error

    def get_sector_rankings(self, n=5):
        if self.sector_error is not None:
            raise self.sector_error
        top, bottom = self.sectors
        return top[:n], bottom[:n]

    def get_hot_stocks(self, n=5):
        if self.stock_error is not None:
            raise self.stock_error
        return self.stocks[:n]


class MarketReviewHotspotServiceTestCase(unittest.TestCase):
    def test_build_markdown_renders_sector_and_stock_sections(self) -> None:
        service = MarketReviewHotspotService(
            manager=_StubManager(
                sectors=(
                    [{"name": "机器人", "change_pct": 6.2}, {"name": "半导体", "change_pct": 4.8}],
                    [],
                ),
                stocks=[
                    {
                        "code": "300308",
                        "name": "中际旭创",
                        "change_pct": 9.91,
                        "amount": 2_300_000_000,
                        "turnover_rate": 8.7,
                    }
                ],
            ),
            top_n=2,
        )

        markdown = service.build_markdown(region="cn", language="zh")

        self.assertIn("### 热门板块", markdown)
        self.assertIn("机器人", markdown)
        self.assertIn("### 热门股票", markdown)
        self.assertIn("中际旭创 (300308)", markdown)
        self.assertIn("成交额 23.0亿", markdown)

    def test_build_markdown_fail_open_when_data_missing(self) -> None:
        service = MarketReviewHotspotService(
            manager=_StubManager(
                sector_error=RuntimeError("sector down"),
                stock_error=RuntimeError("stock down"),
            )
        )

        markdown = service.build_markdown(region="cn", language="zh")

        self.assertEqual(markdown, "")

    def test_build_markdown_skips_non_cn_region(self) -> None:
        service = MarketReviewHotspotService(manager=_StubManager())

        markdown = service.build_markdown(region="us", language="en")

        self.assertEqual(markdown, "")

    def test_build_markdown_skips_invalid_turnover_rate(self) -> None:
        service = MarketReviewHotspotService(
            manager=_StubManager(
                stocks=[
                    {
                        "code": "300308",
                        "name": "中际旭创",
                        "change_pct": 9.91,
                        "amount": 2_300_000_000,
                        "turnover_rate": "-",
                    }
                ]
            )
        )

        markdown = service.build_markdown(region="cn", language="zh")

        self.assertIn("中际旭创 (300308)", markdown)
        self.assertNotIn("换手率", markdown)

    def test_build_markdown_treats_nan_like_missing_numeric_field(self) -> None:
        service = MarketReviewHotspotService(
            manager=_StubManager(
                stocks=[
                    {
                        "code": "300308",
                        "name": "中际旭创",
                        "change_pct": float("nan"),
                        "amount": float("nan"),
                        "turnover_rate": float("nan"),
                    }
                ]
            )
        )

        markdown = service.build_markdown(region="cn", language="zh")

        self.assertIn("中际旭创 (300308)", markdown)
        self.assertIn(": N/A, 成交额 N/A", markdown)
        self.assertNotIn("换手率", markdown)
        self.assertNotIn("nan", markdown.lower())
