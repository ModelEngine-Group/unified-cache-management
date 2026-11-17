# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

CXX ?= g++
BUILD ?= release
NDS_LIB_DIR ?= NDS_lib
BINDIR ?= NDS_bin
TARGET := $(BINDIR)/libndskv.so
SOURCES := NDS3.0/nds_wrapper.cpp NDS3.0/nds_c_adapter.cpp
HEADERS := NDS_include/nds_interface.h ../vllm-uc/NDS_include/NDS_client.h ../vllm-uc/NDS_include/xxhash.h

CPPFLAGS := -INDS_include -I../vllm-uc/NDS_include
COMMON_FLAGS := -W -Wall -Wno-overflow -fPIC -pthread -std=c++17
ifeq ($(BUILD),debug)
    CXXFLAGS := -g -O1 $(COMMON_FLAGS)
else ifeq ($(BUILD),release)
    CXXFLAGS := -O2 -DNDEBUG $(COMMON_FLAGS)
else
    $(error BUILD 只接受 debug 或 release，当前值: $(BUILD))
endif
ifeq ($(NDS_LOG),1)
    CPPFLAGS += -DNDS_LOG
endif

LDFLAGS := -shared -Wl,--no-undefined -Wl,-z,relro,-z,now -L$(NDS_LIB_DIR) -Wl,-rpath,'$$ORIGIN'
LDLIBS := -lndsclient -libverbs -lcrypto -lrt -lz -pthread

.PHONY: all debug release clean

all: $(TARGET)

debug:
	$(MAKE) BUILD=debug all

release:
	$(MAKE) BUILD=release all

$(TARGET): $(SOURCES) $(HEADERS)
	@mkdir -p $(BINDIR)
	@echo "Building $@ ($(BUILD) mode)..."
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(BINDIR) libndskv.so
