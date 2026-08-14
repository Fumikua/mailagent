"""Inbound mail transport adapters and ingestion orchestration."""

from .base import FetchCursor, FetchedMessage, MailGateway

__all__ = ["FetchCursor", "FetchedMessage", "MailGateway"]
