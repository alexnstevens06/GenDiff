"""
LRU Cache Implementation

A Least-Recently-Used (LRU) cache that automatically evicts the oldest
accessed entry when the capacity is exceeded. Internally backed by
`collections.OrderedDict` for O(1) get/put operations.

Usage:
    python lru_cache.py
"""

from collections import OrderedDict
import time


class LRUCache:
    """
    A Least-Recently-Used (LRU) cache with a fixed capacity.

    The cache stores key-value pairs and evicts the least-recently-used
    entry when a new item would exceed the maximum capacity.  Both
    reading (``get``) and writing (``put``) an existing key promote that
    key to the most-recently-used position.

    Internally the data is stored in an :class:`collections.OrderedDict`
    where the *leftmost* item is the least-recently-used and the
    *rightmost* item is the most-recently-used.

    Parameters
    ----------
    capacity : int
        The maximum number of items the cache can hold.  Must be a
        positive integer.

    Raises
    ------
    ValueError
        If *capacity* is not a positive integer.

    Examples
    --------
    >>> cache = LRUCache(2)
    >>> cache.put("a", 1)
    >>> cache.put("b", 2)
    >>> cache.get("a")
    1
    >>> cache.put("c", 3)          # evicts "b"
    >>> cache.get("b") is None
    True
    """

    def __init__(self, capacity: int, ttl: float = 60.0):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity: int = capacity
        self._ttl: float = ttl
        self._cache: OrderedDict = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key) -> object:
        """
        Retrieve the value associated with *key*.

        If the key is present, it is promoted to the most-recently-used
        position before the value is returned.

        Parameters
        ----------
        key : hashable
            The key to look up.

        Returns
        -------
        object or None
            The stored value if the key exists, otherwise ``None``.
        """
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None
            
        # Promote the key to the most-recently-used position.
        self._cache.move_to_end(key)
        return value

    def put(self, key, value) -> None:
        """
        Insert or update a key-value pair in the cache.

        If *key* already exists its value is overwritten and the key is
        promoted to the most-recently-used position.  If the key does
        not exist and the cache is at capacity, the least-recently-used
        entry is evicted first.

        Parameters
        ----------
        key : hashable
            The key to insert or update.
        value : object
            The value to associate with *key*.
        """
        if key in self._cache:
            # Update existing entry and promote it.
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)
        else:
            # Evict the least-recently-used entry if necessary.
            if len(self._cache) >= self._capacity:
                evicted_key, evicted_value = self._cache.popitem(last=False)
                print(f"  [EVICT] key={evicted_key!r} value={evicted_value[0]!r}")
            self._cache[key] = (value, time.time())
            # New items are already appended at the end (most-recently-used).

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of entries currently stored in the cache."""
        return len(self._cache)

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the cache.

        The order reflects the usage order: the first item shown is the
        least-recently-used, and the last item shown is the
        most-recently-used.
        """
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self._cache.items())
        return f"{self.__class__.__name__}(capacity={self._capacity}, {{{items}}})"


# ======================================================================
# Demo / smoke-test
# ======================================================================

if __name__ == "__main__":
    cache = LRUCache(capacity=3)

    operations = [
        ("put", "a", 1),       # 1
        ("put", "b", 2),       # 2
        ("put", "c", 3),       # 3
        ("get", "a"),          # 4 – promotes "a" → order: b, c, a
        ("put", "d", 4),       # 5 – evicts "b" → order: c, a, d
        ("get", "b"),          # 6 – miss
        ("put", "e", 5),       # 7 – evicts "c" → order: a, d, e
        ("put", "a", 10),      # 8 – updates "a", promotes it → order: d, e, a
    ]

    print(f"Created {cache!r}\n")

    for idx, op in enumerate(operations, start=1):
        action = op[0]
        key = op[1]

        if action == "put":
            value = op[2]
            print(f"{idx}. put(key={key!r}, value={value!r})")
            cache.put(key, value)
        elif action == "get":
            print(f"{idx}. get(key={key!r})")
            result = cache.get(key)
            print(f"   → returned {result!r}")

        print(f"   State: {cache!r}  (len={len(cache)})")
        print()