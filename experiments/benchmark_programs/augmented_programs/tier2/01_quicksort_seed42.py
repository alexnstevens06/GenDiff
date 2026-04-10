"""
This module provides two implementations of quicksort:
- An in-place version that mutates the input list.
- A pure functional version that returns a new sorted list without modifying the input.

Both implementations handle edge cases: empty list, single element, and all-duplicates.
"""

import random


def quicksort_inplace(arr: list) -> None:
    """Sort the list in ascending order using in-place quicksort.

    The algorithm uses the Lomuto partition scheme with random pivot selection
    to avoid worst-case behavior on already sorted or nearly sorted inputs.
    The list is sorted in-place; the function returns None.

    Args:
        arr: List of comparable items to be sorted. The list is mutated.

    Returns:
        None

    Edge cases:
        - Empty list: no change.
        - Single element: no change.
        - All duplicates: sorted correctly (order of equal elements may change).
    """

    def _partition(low: int, high: int) -> int:
        """Partition the subarray arr[low:high+1] using the Lomuto scheme.

        A random element is chosen as pivot and swapped to the last position.
        Elements <= pivot are moved to the left, > pivot to the right.
        The final index of the pivot is returned.

        Args:
            low: Leftmost index of the subarray.
            high: Rightmost index of the subarray.

        Returns:
            The index where the pivot is placed after partitioning.
        """
        # Random pivot selection to avoid worst-case scenarios
        pivot_idx = random.randint(low, high)
        arr[high], arr[pivot_idx] = arr[pivot_idx], arr[high]
        pivot = arr[high]

        i = low - 1  # index of the smaller element
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        # Place pivot in its correct position
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def _insertion_sort(low: int, high: int) -> None:
        """Helper to sort small subarrays using insertion sort."""
        for i in range(low + 1, high + 1):
            key = arr[i]
            j = i - 1
            while j >= low and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def _quicksort(low: int, high: int) -> None:
        """Recursively sort the subarray arr[low:high+1]."""
        if low < high:
            # Hybrid optimization: use insertion sort for small arrays
            if high - low < 10:
                _insertion_sort(low, high)
            else:
                p = _partition(low, high)
                _quicksort(low, p - 1)
                _quicksort(p + 1, high)

    # Start the recursion; handles empty and single-element lists automatically
    _quicksort(0, len(arr) - 1)


def quicksort_functional(lst: list) -> list:
    """Return a new sorted list using pure functional quicksort.

    The algorithm uses a three-way partition (less, equal, greater) to handle
    duplicates efficiently and avoid infinite recursion. The pivot is chosen
    as the first element of the list for simplicity.

    Args:
        lst: List of comparable items to be sorted.

    Returns:
        A new list containing the elements of `lst` in ascending order.

    Edge cases:
        - Empty list: returns an empty list.
        - Single element: returns a list with that element.
        - All duplicates: returns a list with all elements (order among equals
          is preserved because the equal partition retains original order).
    """
    if len(lst) <= 1:
        # Return a copy to maintain immutability
        return lst[:]

    pivot = lst[0]
    less = [x for x in lst if x < pivot]
    equal = [x for x in lst if x == pivot]
    greater = [x for x in lst if x > pivot]

    return quicksort_functional(less) + equal + quicksort_functional(greater)


if __name__ == "__main__":
    import time

    # Generate a random list of 20 integers, with RNG seeded for reproducibility
    random.seed(42)
    original = [random.randint(0, 100) for _ in range(20)]
    print("Original list:", original)

    # In-place sort (use a copy to keep original unchanged)
    inplace_list = original[:]
    start_inplace = time.perf_counter()
    quicksort_inplace(inplace_list)
    end_inplace = time.perf_counter()
    print("\nIn-place sorted list:", inplace_list)
    print(f"In-place quicksort took {end_inplace - start_inplace:.6f} seconds")

    # Functional sort (original list is not modified)
    start_functional = time.perf_counter()
    sorted_functional = quicksort_functional(original)
    end_functional = time.perf_counter()
    print("\nFunctional sorted list:", sorted_functional)
    print(f"Functional quicksort took {end_functional - start_functional:.6f} seconds")

    # Verify correctness by comparing with Python's built‑in sort
    reference = sorted(original)
    if inplace_list == sorted_functional == reference:
        print("\nBoth sorts produce the same correct result.")
    else:
        print("\nWARNING: Sorting results differ!")