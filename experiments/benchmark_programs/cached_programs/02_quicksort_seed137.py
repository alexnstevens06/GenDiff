"""
quicksort.py

A self-contained Python module implementing the quicksort algorithm in two
styles:

  1. In-place quicksort — mutates the input list (Hoare partition scheme).
  2. Pure functional quicksort — returns a new sorted list, leaves the
     input untouched.

Both implementations correctly handle edge cases: empty lists,
single-element lists, and lists of all-duplicate values.
"""

import random
import time


# ---------------------------------------------------------------------------
# In-place quicksort (Hoare partition scheme)
# ---------------------------------------------------------------------------

def quicksort_inplace(arr, low=0, high=None):
    """Sort a list in-place using quicksort with Hoare's partition scheme.

    This function mutates the input list and returns None, following the
    convention of Python's built-in ``list.sort()``.

    The middle element of the current subarray is chosen as the pivot,
    which avoids worst-case O(n²) behaviour on already-sorted input.

    Args:
        arr:  The list to be sorted (modified in-place).
        low:  Starting index of the subarray to sort (default 0).
        high: Ending index of the subarray to sort  (default ``len(arr)-1``).

    Returns:
        None.  The list is sorted in-place.

    Edge cases:
        * Empty list — the initial ``low < high`` check is False, so nothing
          happens.
        * Single element — base case; no swaps needed.
        * All duplicates — Hoare's scheme distributes equal elements across
          both partitions, but the subarray still shrinks each recursive step
          so convergence is guaranteed.
    """
    if high is None:
        high = len(arr) - 1

    # Base case: a subarray of 0 or 1 elements is already sorted
    if low < high:
        pivot_idx = _hoare_partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_idx)        # sort left partition
        quicksort_inplace(arr, pivot_idx + 1, high)   # sort right partition


def _hoare_partition(arr, low, high):
    """Partition a subarray around a pivot using Hoare's scheme.

    Elements ≤ pivot end up in ``arr[low..j]`` and elements ≥ pivot end up
    in ``arr[j+1..high]``.  The two partitions may each still contain values
    equal to the pivot, but every element is on the correct side relative
    to the pivot value.

    Args:
        arr:  The list containing the subarray to partition.
        low:  Starting index of the subarray.
        high: Ending index of the subarray.

    Returns:
        The index ``j`` that marks the boundary between the two partitions.
    """
    # Choose the middle element as pivot for balanced partitions
    pivot = arr[(low + high) // 2]

    i = low - 1    # left scanner — starts just before the subarray
    j = high + 1   # right scanner — starts just after the subarray

    while True:
        # Advance i rightward until arr[i] >= pivot
        i += 1
        while arr[i] < pivot:
            i += 1

        # Advance j leftward until arr[j] <= pivot
        j -= 1
        while arr[j] > pivot:
            j -= 1

        # If scanners have met or crossed, the partition is complete
        if i >= j:
            return j

        # Swap the two out-of-place elements and continue
        arr[i], arr[j] = arr[j], arr[i]


# ---------------------------------------------------------------------------
# Pure functional quicksort
# ---------------------------------------------------------------------------

def quicksort_functional(arr):
    """Return a new sorted list using a pure-functional quicksort.

    The input is never modified.  Elements are partitioned into three
    groups via list comprehensions (less / equal / greater) so that
    duplicate values are handled correctly and efficiently.

    Args:
        arr: An iterable (typically a list) of comparable elements.

    Returns:
        A **new** list containing all elements of ``arr`` in non-decreasing
        order.

    Edge cases:
        * Empty list — returns ``[]``.
        * Single element — returns a new one-element list.
        * All duplicates — the ``equal`` partition captures every element,
          both recursive calls receive empty lists, and the result is
          returned immediately after a single level of recursion.
    """
    # Base case: 0 or 1 elements are already sorted
    if len(arr) <= 1:
        return list(arr)          # return a *new* list (preserves immutability)

    # Choose the middle element as pivot for balanced partitions
    pivot = arr[len(arr) // 2]

    # Three-way partition avoids infinite recursion on all-duplicate input
    less    = [x for x in arr if x < pivot]
    equal   = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]

    # Recursively sort the sub-partitions and concatenate
    return quicksort_functional(less) + equal + quicksort_functional(greater)


# ---------------------------------------------------------------------------
# Demonstration / simple test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Generate a reproducible random list --------------------------------
    random.seed(42)
    original = [random.randint(1, 100) for _ in range(20)]
    original_snapshot = original.copy()   # keep a copy for later verification

    print("=" * 60)
    print("Quicksort Demonstration")
    print("=" * 60)
    print(f"\nOriginal list ({len(original)} elements):")
    print(f"  {original}\n")

    # ---- In-place quicksort -------------------------------------------------
    arr_inplace = original.copy()         # work on a copy so original survives
    t0 = time.time()
    quicksort_inplace(arr_inplace)
    t1 = time.time()

    print("In-place quicksort result:")
    print(f"  {arr_inplace}")
    print(f"  Elapsed: {t1 - t0:.6f} seconds\n")

    # ---- Functional quicksort ------------------------------------------------
    t0 = time.time()
    arr_functional = quicksort_functional(original)
    t1 = time.time()

    print("Functional quicksort result:")
    print(f"  {arr_functional}")
    print(f"  Elapsed: {t1 - t0:.6f} seconds\n")

    # ---- Verify correctness --------------------------------------------------
    reference = sorted(original_snapshot)

    print("-" * 60)
    print("Verification")
    print("-" * 60)
    print(f"  In-place matches sorted():      {arr_inplace == reference}")
    print(f"  Functional matches sorted():     {arr_functional == reference}")
    print(f"  Both implementations agree:      {arr_inplace == arr_functional}")
    print(f"  Original list unchanged:         "
          f"{original == original_snapshot}\n")

    # ---- Edge-case demonstrations --------------------------------------------
    print("-" * 60)
    print("Edge-case tests")
    print("-" * 60)

    # Empty list
    empty_ip = []
    quicksort_inplace(empty_ip)
    empty_fn = quicksort_functional([])
    print(f"\n  Empty list:")
    print(f"    In-place:     {empty_ip}")
    print(f"    Functional:   {empty_fn}")

    # Single element
    single_ip = [42]
    single_backup = single_ip.copy()
    quicksort_inplace(single_ip)
    single_fn = quicksort_functional(single_backup)
    print(f"\n  Single element [42]:")
    print(f"    In-place:     {single_ip}")
    print(f"    Functional:   {single_fn}")

    # All duplicates
    dup_ip = [7, 7, 7, 7, 7, 7, 7]
    dup_backup = dup_ip.copy()
    quicksort_inplace(dup_ip)
    dup_fn = quicksort_functional(dup_backup)
    print(f"\n  All duplicates [7]*7:")
    print(f"    In-place:     {dup_ip}")
    print(f"    Functional:   {dup_fn}")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)