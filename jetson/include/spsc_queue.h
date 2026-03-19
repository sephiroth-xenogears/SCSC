#pragma once
// ============================================================================
// Phase 7: Lock-free SPSC (Single Producer Single Consumer) Queue
// Used for inter-thread data passing per §7.2
// ============================================================================

#include <atomic>
#include <cstddef>

namespace phase7 {

template <typename T, size_t Capacity = 16>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
public:
    bool tryPush(const T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) & MASK;
        if (next == tail_.load(std::memory_order_acquire)) return false; // full
        buffer_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool tryPop(T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return false; // empty
        item = buffer_[tail];
        tail_.store((tail + 1) & MASK, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

private:
    static constexpr size_t MASK = Capacity - 1;
    T buffer_[Capacity];
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

} // namespace phase7
