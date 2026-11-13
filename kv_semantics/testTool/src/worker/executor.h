/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

#include <string>

class Client;

/**
 * 客户端测试执行器：基于已绑定的 kv client 执行功能验证流程，
 * 当前为单线程基础版本（注册→写入→回读→删除→注销），
 * 后续将扩展为多线程并发提交（线程数等参数取自配置）。
 */
class Executor {
public:
    /** 绑定已初始化的 kv client（不拥有，生命周期由调用方保证） */
    void SetClient(Client *client);

    /** 执行一轮基础功能验证（注册→写前查→写入→写后查→回读比对→删除→删后查→注销）；失败时 err 说明首次失败的步骤 */
    bool Run(std::string &err);

private:
    Client *client_{nullptr};
};
