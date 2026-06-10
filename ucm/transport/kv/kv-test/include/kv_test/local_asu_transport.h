#pragma once

#include <string>
#include "asu_client/asu_client.h"

namespace UC::KVTest {

UC::ASU::TransportFactory CreateLocalAsuTransportFactory(std::string storeRoot);

}  // namespace UC::KVTest
