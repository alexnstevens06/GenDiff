"""
lru_cache.py

A self-contained implementation of an LRU (Least-Recently-Used) cache
using collections.OrderedDict from the standard library.

Usage:
    python lru_cache.py
"""

from collections import OrderedDict
from typing import Any
import threading


class LRUCache:
    """
    A Least-Recently-Used (LRU) cache that evicts the oldest accessed
    entry when the cache exceeds its capacity.

    Internally backed by collections.OrderedDict, which maintains
    insertion order. On every access (get) or insertion (put), the
    relevant key is moved to the end of the order, making it the
    "most recently used". When eviction is necessary, the item at
    the front (least recently used) is removed.

    Attributes:
        capacity (int): Maximum number of items the cache can hold.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialise the LRU cache with a given capacity.

        Args:
            capacity: Maximum number of key-value pairs the cache stores.
                      Must be a positive integer.

        Raises:
            ValueError: If capacity is less than 1.
        """
        if capacity < 1:
            raise ValueError(f"Capacity must be >= 1, got {capacity}")
        self._capacity: int = capacity
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        """Return the maximum capacity of the cache."""
        return self._capacity

    def get(self, key: Any) -> Any:
        """
        Retrieve the value associated with *key*, marking it as most recently used.

        If the key exists, it is moved to the end of the internal order
        so it becomes the most-recently-used entry.

        Args:
            key: The key to look up.

        Returns:
            The value associated with *key* if it exists, otherwise None.
        """
        with self._lock:
            if key not in self._cache:
                return None
            # Move the key to the end to mark it as most recently used
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: Any, value: Any) -> None:
        """
        Insert or update a key-value pair in the cache, marking it as most recently used.

        If the key already exists, its value is updated and it is moved to the
        end (most recently used). If the key is new and the cache is at capacity,
        the least-recently-used entry (the front of the order) is evicted first.

        Args:
            key:   The key to insert or update.
            value: The value to associate with the key.
        """
        with self._lock:
            if key in self._cache:
                # Update existing entry and move to end (most recently used)
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                # Evict LRU entry if at capacity
                if len(self._cache) >= self._capacity:
                    evicted_key = next(iter(self._cache))
                    evicted_value = self._cache.pop(evicted_key)
                    print(f"  ↳ EVICTION: removed ({evicted_key!r}: {evicted_value!r})")
                self._cache[key] = value  # new entries go to the end by default

    def __len__(self) -> int:
        """Return the current number of items in the cache."""
        return len(self._cache)

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the cache.

        Items are shown in order from least-recently-used (left) to
        most-recently-used (right).
        """
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self._cache.items())
        return f"LRUCache(capacity={self._capacity}, {{{items}}})"


# ──────────────────────────────────────────────────────────────────────────────
# Demo / smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cache = LRUCache(capacity=3)
    print(f"Created {cache}\n")

    operations: list[tuple[str, tuple, str]] = [
        # (method_name, args, description)
        ("put", ("A", 1), "Insert A:1          → cache fills slot 1/3"),
        ("put", ("B", 2), "Insert B:2          → cache fills slot 2/3"),
        ("put", ("C", 3), "Insert C:3          → cache fills slot 3/3 (full)"),
        ("get", ("A",),  "Get A               → A becomes most recent again"),
        ("put", ("D", 4), "Insert D:4          → evicts B (LRU because A was just accessed)"),
        ("get", ("B",),  "Get B (miss)        → B was evicted, returns None"),
        ("put", ("E", 5), "Insert E:5          → evicts C (LRU after previous ops)"),
        ("put", ("A", 9), "Update A:9          → A already present, just updated; no eviction"),
    ]

    for method_name, args, description in operations:
        method = getattr(cache, method_name)
        result = method(*args)
        display_result = f" → returned {result!r}" if result is not None else ""
        print(f"{description}{display_result}")
        print(f"  State: {cache}  (size={len(cache)})")
        print()