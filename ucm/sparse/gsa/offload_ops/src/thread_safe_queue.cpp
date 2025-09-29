#include "thread_safe_queue.h"

ThreadSafeQueue::ThreadSafeQueue() : m_stopped(false) {}

void ThreadSafeQueue::push(CopyInfo value) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_queue.push(std::move(value));
    m_condVar.notify_one();
}

CopyInfo ThreadSafeQueue::pop() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_condVar.wait(lock, [this] { 
        return !m_queue.empty() || m_stopped; 
    });
    CopyInfo value = std::move(m_queue.front());
    m_queue.pop();
    return value;
}

size_t ThreadSafeQueue::size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.size();
}

bool ThreadSafeQueue::empty() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.empty();
}

void ThreadSafeQueue::stop() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stopped = true;
    m_condVar.notify_all();
}

void ThreadSafeQueue::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    while (!m_queue.empty()) {
        m_queue.pop();
    }
}