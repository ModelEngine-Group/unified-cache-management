/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "status/status.h"

namespace UC::Transfer {

/** Wire limit of one frame's payload. */
inline constexpr std::uint32_t kUdsMaxFrameSize = 64u * 1024u * 1024u;

/** Maximum number of descriptors one frame may carry over SCM_RIGHTS. */
inline constexpr std::size_t kUdsMaxFdsPerFrame = 64;

/**
 * @brief One connected endpoint of a Unix-domain socket stream, speaking the
 *        framed control-plane protocol: every message travels as
 *        [4B little-endian length][payload], optionally accompanied by file
 *        descriptors attached to the frame's first byte via SCM_RIGHTS.
 *
 * The connection is a dumb byte-and-fd pipe: it understands neither control
 * messages nor roles -- all message semantics live in the control codec on
 * top. Errors surface as error Status values only (no timeouts, no
 * self-termination, no monitoring); a peer that closed the socket yields an
 * error Status from the next operation.
 *
 * @note Threading contract: not thread-safe. Callers must serialize the
 *       send/receive operations of one connection (single consumer, matching
 *       the Store control-plane contract).
 * @note Partial writes of oversized frames are transparently completed; the
 *       ancillary descriptors are handed to the kernel together with the
 *       first byte, so they are delivered exactly once even when the frame
 *       spans multiple sendmsg calls.
 */
class UdsConnection {
public:
    UdsConnection() noexcept = default;
    ~UdsConnection();
    UdsConnection(UdsConnection&& other) noexcept;
    UdsConnection& operator=(UdsConnection&& other) noexcept;
    UdsConnection(const UdsConnection&) = delete;
    UdsConnection& operator=(const UdsConnection&) = delete;

    /**
     * @brief Connects to a listening endpoint at the given filesystem path.
     *
     * @param path Filesystem path of the server socket (bound by UdsListener).
     *
     * @return Expected an open connection on success; error Status (OsApiError)
     *         when the socket path does not exist, nobody is listening, or a
     *         socket-level error occurs. Retrying a not-yet-started peer is
     *         the caller's business (backoff policy belongs to bootstrap).
     */
    [[nodiscard]] static Expected<UdsConnection> Connect(const std::string& path);

    /** @brief Tells whether the endpoint holds an open descriptor. */
    [[nodiscard]] bool IsValid() const noexcept { return fd_ >= 0; }

    /** @brief The endpoint's descriptor, for callers that multiplex it in
     *         their own event loop. Read-only: ownership stays here. */
    [[nodiscard]] int Fd() const noexcept { return fd_; }

    /**
     * @brief Sends one frame: [4B length][payload] without descriptors.
     */
    Status SendFrame(const std::vector<std::byte>& payload);

    /**
     * @brief Sends one frame carrying file descriptors via SCM_RIGHTS.
     *
     * The descriptors are attached to the frame's first byte; the receiver
     * collects them from the ReceiveFrame call that delivers that byte.
     * Passing an empty descriptor list degrades to SendFrame.
     *
     * @param payload Frame payload, at most kUdsMaxFrameSize bytes.
     * @param fds     Descriptors to transfer; the receiver obtains its own
     *                duplicated copies. Invalid descriptors yield an
     *                OsApiError from sendmsg.
     */
    Status SendFrameWithFds(const std::vector<std::byte>& payload, const std::vector<int>& fds);

    /**
     * @brief Receives one frame and the descriptors arriving with it.
     *
     * @param payload      [out] Receiving buffer; capacity is reused across
     *                     calls. Holds the frame payload on success, and is
     *                     unspecified after an error.
     * @param receivedFds  [out] Descriptors duplicated into this process by
     *                     SCM_RIGHTS, in arrival order; empty when the frame
     *                     carried none. Ownership of the descriptors passes
     *                     to the caller.
     *
     * @return Status::Ok() after a complete frame arrived; InvalidParam when
     *         the frame length exceeds kUdsMaxFrameSize or the ancillary data
     *         overflows the receive budget; OsApiError on socket errors; a
     *         General error when the peer closed the connection.
     */
    Status ReceiveFrame(std::vector<std::byte>& payload, std::vector<int>& receivedFds);

    /**
     * @brief Closes the endpoint. Idempotent; a closed endpoint rejects
     *        further operations with InvalidParam.
     */
    void Close() noexcept;

private:
    friend class UdsListener;
    explicit UdsConnection(int fd) noexcept : fd_{fd} {}

    int fd_{-1};
};

/**
 * @brief Listening endpoint of a Unix-domain socket stream.
 *
 * Owns socket/bind/listen over one filesystem path. The listener performs no
 * cleanup of the socket file itself: stale-artifact reclamation belongs to
 * the bootstrap leader's GC, so bind fails with an OsApiError when the path
 * is still occupied.
 *
 * @note Threading contract: not thread-safe; callers serialize Listen and
 *       Accept.
 */
class UdsListener {
public:
    UdsListener() noexcept = default;
    ~UdsListener();
    UdsListener(UdsListener&& other) noexcept;
    UdsListener& operator=(UdsListener&& other) noexcept;
    UdsListener(const UdsListener&) = delete;
    UdsListener& operator=(const UdsListener&) = delete;

    /**
     * @brief Creates the socket, binds it to the given path and starts
     *        listening.
     *
     * @param path Absolute filesystem path of the server socket; shorter
     *             than the sockaddr_un limit (107 bytes).
     *
     * @return Status::Ok() on success; InvalidParam for an empty or
     *         over-long path; OsApiError when bind or listen fails (e.g. the
     *         path already exists -- see the class note about GC ownership).
     */
    Status Listen(const std::string& path);

    /**
     * @brief Blocks until a connection arrives and returns it.
     *
     * @return Expected an open UdsConnection on success; OsApiError when
     *         accept fails. No timeout: waiting bounds belong to the caller.
     */
    [[nodiscard]] Expected<UdsConnection> Accept();

    /** @brief Tells whether the listener holds an open descriptor. */
    [[nodiscard]] bool IsValid() const noexcept { return fd_ >= 0; }

    /** @brief The listening socket's descriptor, for callers that multiplex
     *         it in their own event loop. Read-only: ownership stays here. */
    [[nodiscard]] int Fd() const noexcept { return fd_; }

    /** @brief Filesystem path the listener is bound to, empty when not listening. */
    [[nodiscard]] const std::string& Path() const noexcept { return path_; }

    /**
     * @brief Closes the listener. Idempotent. Does not unlink the socket
     *        file (GC ownership stays with the bootstrap leader).
     */
    void Close() noexcept;

private:
    int fd_{-1};
    std::string path_;
};

}  // namespace UC::Transfer
