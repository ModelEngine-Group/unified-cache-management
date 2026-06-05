#pragma once

#include <fmt/format.h>

namespace fmt {
#if FMT_VERSION < 110000
template <typename E>
constexpr auto underlying(E e) noexcept -> std::underlying_type_t<E>
{ return static_cast<std::underlying_type_t<E>>(e); }
#endif
}  // namespace fmt
