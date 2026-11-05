CC := /usr/bin/g++
LIB_TARGET := libndskv.so
BINDIR := NDS_bin
SRCS_NDS := NDS3.0/nds_wrapper.cpp NDS3.0/nds_c_adapter.cpp
HEADERS_NDS := NDS_include/nds_interface.h
# NDS3.0 源码不依赖 pybind11（pybind 扩展在 ../../src/main.cpp，由根目录 setup.py
# 单独编译），此处不再注入 pybind includes，C++ 库编译不依赖 python 环境。
INCLUDE_NDS := -INDS_include
# --no-undefined：链接期校验全部符号均有定义，缺符号当场报错而非运行时崩；
# -z relro,-z now：full RELRO + 加载时立即绑定，依赖缺失/版本错配 fail-fast。
LDFLAGS := -shared -Wl,--no-undefined -Wl,-z,relro,-z,now -libverbs -lcrypto -lndsclient -lrt -Wl,-rpath,'$$ORIGIN/../NDS_lib'
MY_DEF ?=

ifeq ($(NDS_LOG), 1)
	MY_DEF += -DNDS_LOG
endif

ifeq ($(MAKECMDGOALS),)
	BUILD_TYPE := debug
else ifneq (,$(filter debug release,$(MAKECMDGOALS)))
	BUILD_TYPE := $(MAKECMDGOALS)
endif

ifeq ($(BUILD_TYPE),release)
	CFLAGS := -O1 -W -Wall -Wno-overflow -fvisibility=hidden -fno-rtti -DNDEBUG -fPIC -pthread -mllvm -fla -mllvm -bcf -mllvm \
		-sub -mllvm -split -mllvm -sobf $(INCLUDE_NDS) -L./NDS_lib
else
	CFLAGS := -g -O1 -W -Wall -Wno-overflow -fPIC -pthread $(INCLUDE_NDS) -L./NDS_lib
endif

.PHONY: all debug release clean

all: $(LIB_TARGET)
	rm -rf $(BINDIR)
	mkdir $(BINDIR)
	mv $(LIB_TARGET) $(BINDIR)

debug: all
release: all

$(LIB_TARGET): $(SRCS_NDS) $(HEADERS_NDS)
	@echo "Building Shared Library $@ ($(BUILD_TYPE) mode)..."
	$(CC) $(CFLAGS) $(MY_DEF) $(SRCS_NDS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BINDIR)/*
	rm -rf $(LIB_TARGET)