# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

CXX ?= g++
BUILD ?= release
AGENT_MODE ?= auto
AGENT_LIB_DIR ?= agent/release
BINDIR ?= NDS_bin
TARGET := $(BINDIR)/libndsclient.so

SOURCES := $(wildcard NDS3.0/*.c)
CPPFLAGS := -INDS_include
COMMON_FLAGS := -W -Wall -Wno-overflow -fPIC -pthread -std=c++17
ifeq ($(BUILD),debug)
    CXXFLAGS := -g -O0 $(COMMON_FLAGS)
else ifeq ($(BUILD),release)
    CXXFLAGS := -O2 -DNDEBUG $(COMMON_FLAGS)
else
    $(error BUILD 只接受 debug 或 release，当前值: $(BUILD))
endif
ifeq ($(NDS_LOG),1)
    CPPFLAGS += -DNDS_LOG
endif

# AGENT_MODE=auto：两个厂商库齐全才启用集群 agent，否则使用非集群 stub。
RESOLVED_AGENT_MODE := $(AGENT_MODE)
ifeq ($(AGENT_MODE),auto)
    ifneq ($(wildcard $(AGENT_LIB_DIR)/libkv_agent.so),)
        ifeq ($(wildcard $(AGENT_LIB_DIR)/libceph-common.so.2),)
            $(error 找到 libkv_agent.so，但缺少 $(AGENT_LIB_DIR)/libceph-common.so.2)
        endif
        RESOLVED_AGENT_MODE := release
    else
        ifneq ($(wildcard $(AGENT_LIB_DIR)/libceph-common.so.2),)
            $(error 找到 libceph-common.so.2，但缺少 $(AGENT_LIB_DIR)/libkv_agent.so)
        endif
        RESOLVED_AGENT_MODE := none
    endif
endif

# 兼容旧的 AGENT_MOCK=true/false 调用；新代码请使用 AGENT_MODE。
ifdef AGENT_MOCK
    ifeq ($(AGENT_MOCK),true)
        RESOLVED_AGENT_MODE := mock
    else ifeq ($(AGENT_MOCK),false)
        RESOLVED_AGENT_MODE := release
    else
        $(error AGENT_MOCK 只接受 true 或 false，当前值: $(AGENT_MOCK))
    endif
endif

LDFLAGS := -shared -Wl,--no-undefined -Wl,-z,relro,-z,now -Wl,-rpath,'$$ORIGIN'
LDLIBS := -libverbs -lcrypto -lz -lpthread
CPPFLAGS += -DXXH_STATIC_LINKING_ONLY -DXXH_INLINE_ALL

ifeq ($(RESOLVED_AGENT_MODE),release)
    CPPFLAGS += -Iagent/release
    LDFLAGS += -L$(AGENT_LIB_DIR) -Wl,-rpath-link,$(abspath $(AGENT_LIB_DIR))
    LDLIBS += -l:libkv_agent.so
    COPY_AGENT_LIBS := libkv_agent.so libceph-common.so.2
else ifeq ($(RESOLVED_AGENT_MODE),mock)
    CPPFLAGS += -Iagent/mock
    SOURCES += $(wildcard agent/mock/*.c)
else ifeq ($(RESOLVED_AGENT_MODE),none)
    CPPFLAGS += -Iagent/release
    SOURCES += agent/stub/KVClientAgent.c
else
    $(error AGENT_MODE 只接受 auto、release、mock 或 none，当前值: $(AGENT_MODE))
endif

.PHONY: all debug release clean

all: $(TARGET)

debug:
	$(MAKE) BUILD=debug AGENT_MODE=$(AGENT_MODE) AGENT_LIB_DIR=$(AGENT_LIB_DIR) all

release:
	$(MAKE) BUILD=release AGENT_MODE=$(AGENT_MODE) AGENT_LIB_DIR=$(AGENT_LIB_DIR) all

$(TARGET): $(SOURCES) NDS_include/xxhash.h
	@mkdir -p $(BINDIR)
	@echo "Building $@ ($(BUILD) mode, AGENT_MODE=$(RESOLVED_AGENT_MODE))..."
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) -o $@ $(LDFLAGS) $(LDLIBS)
ifeq ($(RESOLVED_AGENT_MODE),release)
	cp -a $(addprefix $(AGENT_LIB_DIR)/,$(COPY_AGENT_LIBS)) $(BINDIR)/
endif

clean:
	rm -rf $(BINDIR) libndsclient.so libclient_agent.so
