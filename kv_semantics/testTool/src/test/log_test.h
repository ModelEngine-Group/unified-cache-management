/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#pragma once

class ConfigUtil;

/**
 * 日志模块功能自测（临时，正式功能开发完成后随 src/test/ 目录整体删除）：
 * 输出各级别样例与批量日志触发滚动，落盘后校验文件内容，全部通过返回 true。
 * @param config 全局配置实例（显式传入，日志目录与文件名等参数由此读取）
 */
bool RunLogTest(const ConfigUtil &config);
