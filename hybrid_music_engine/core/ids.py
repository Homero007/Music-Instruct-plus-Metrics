from __future__ import annotations

import re
import uuid
from datetime import datetime


def slugify(value: str, fallback: str = "item") -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return text or fallback


def create_id(label: str, prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{stamp}-{prefix}-{slugify(label)}-{suffix}"
