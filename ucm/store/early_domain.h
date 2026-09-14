#ifndef UCM_EARLY_DOMAIN_H
#define UCM_EARLY_DOMAIN_H
#include <string>
#include "ucmstore_v1.h"

namespace UC::Early {
// Optional capability of the existing Early store. No resource or shared index.
struct Domain {
    std::string backing;
    size_t nodes{0}, reservable{0}, shardBytes{0}, blockBytes{0};
};
class DomainProvider {
public:
    virtual ~DomainProvider() = default;
    virtual Domain DescribeDomain() = 0;
};
}  // namespace UC::Early
#endif
