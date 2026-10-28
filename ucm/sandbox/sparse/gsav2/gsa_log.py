"""
================================================================================
                              GSA 日志系统使用说明
================================================================================

一、日志类型
--------------------------------------------------------------------------------
GSALogger 日志系统支持以下三种日志类型：

  1. 性能日志 (performance)
     - 用于记录性能相关的信息，如推理延迟、吞吐量、缓存命中率等
     - 方法前缀: performance_

  2. 精度日志 (accuracy)
     - 用于记录精度相关的信息，如计算结果精度、误差分析等
     - 方法前缀: accuracy_

  3. 日常信息日志 (normal)
     - 用于记录日常运行信息，如调试信息、状态变更等
     - 方法前缀: normal_

  4. 所有日志 (all)
     - 同时记录上述所有类型的日志，分别写入不同的日志文件
     - 默认选项

二、日志级别
--------------------------------------------------------------------------------
每种日志类型都支持以下四个日志级别：

  - DEBUG   (10): 调试信息
  - INFO    (20): 一般信息
  - WARNING (30): 警告信息（包含错误日志）
  - ERROR   (40): 错误信息

三、使用方法
--------------------------------------------------------------------------------

  1. 设置环境变量 GSA_LOG_TYPE 选择日志类型（可选，默认 all）:

       export GSA_LOG_TYPE=performance   # 只记录性能日志
       export GSA_LOG_TYPE=accuracy      # 只记录精度日志
       export GSA_LOG_TYPE=normal        # 只记录日常信息日志
       export GSA_LOG_TYPE=all           # 记录所有日志（默认）

  2. 在代码中使用日志记录方法:

       # 性能日志
       self.log.performance_debug("这是一条性能调试信息")
       self.log.performance_info("这是一条性能信息")
       self.log.performance_warning("这是一条性能警告信息")
       self.log.performance_error("这是一条性能错误信息")

       # 精度日志
       self.log.accuracy_debug("这是一条精度调试信息")
       self.log.accuracy_info("这是一条精度信息")
       self.log.accuracy_warning("这是一条精度警告信息")
       self.log.accuracy_error("这是一条精度错误信息")

       # 日常信息日志
       self.log.normal_debug("这是一条日常调试信息")
       self.log.normal_info("这是一条日常信息")
       self.log.normal_warning("这是一条日常警告信息")
       self.log.normal_error("这是一条日常错误信息")

  3. 兼容原有接口（默认写入 normal 类型）:

       self.log.debug("调试信息")    # 等同于 self.log.normal_debug()
       self.log.info("一般信息")     # 等同于 self.log.normal_info()
       self.log.warning("警告信息")  # 等同于 self.log.normal_warning()
       self.log.error("错误信息")    # 等同于 self.log.normal_error()

四、日志文件
--------------------------------------------------------------------------------
日志文件保存在 gsa_log/GLM/pyLog/ 目录下，文件名格式为:

  log_for_{dp_rank}_tp_{tp_rank}_{log_type}.log

例如（当 GSA_LOG_TYPE=all 时）:
  - log_for_0_tp_0_performance.log  (性能日志)
  - log_for_0_tp_0_accuracy.log     (精度日志)
  - log_for_0_tp_0_normal.log       (日常信息日志)

五、代码示例
--------------------------------------------------------------------------------

  # 初始化日志（根据环境变量选择日志类型）
  log_type = os.getenv("GSA_LOG_TYPE", GSALogger.LOG_TYPE_ALL)
  self.log = GSALogger(
      log_dir="gsa_log/GLM/pyLog",
      dp_rank=self.dp_rank,
      tp_rank=self.rank,
      log_type=log_type,
      show_time=False
  )

  # 记录性能日志
  self.log.performance_info(f"Step {step} 推理延迟: {latency}ms")

  # 记录精度日志
  self.log.accuracy_info(f"计算结果误差: {error}")

  # 记录日常日志（调试信息）
  self.log.normal_debug(f"缓存状态: {cache_status}")

  # 使用兼容接口
  self.log.info(f"请求处理完成: {req_id}")

================================================================================
"""

import os
from datetime import datetime, timezone


class Logger:
    # 日志级别
    LEVEL_DEBUG = 10
    LEVEL_INFO = 20
    LEVEL_WARNING = 30
    LEVEL_ERROR = 40

    LEVEL_NAMES = {
        LEVEL_DEBUG: "DEBUG",
        LEVEL_INFO: "INFO",
        LEVEL_WARNING: "WARNING",
        LEVEL_ERROR: "ERROR",
    }

    def __init__(
        self, log_file="logs/app.log", enabled=True, level=LEVEL_INFO, show_time=True
    ):
        """
        :param log_file: 日志文件路径
        :param enabled: 日志总开关
        :param level: 最低日志级别
        :param show_time: 是否打印时间
        """
        self.log_file = log_file
        self.enabled = enabled
        self.level = level
        self.show_time = show_time

        # 日志文件夹不存在则创建
        log_dir = os.path.dirname(os.path.abspath(self.log_file))
        os.makedirs(log_dir, exist_ok=True)

    def set_enabled(self, enabled: bool):
        """设置日志开关"""
        self.enabled = enabled

    def set_level(self, level):
        """设置最低日志级别"""
        self.level = level

    def set_show_time(self, show_time: bool):
        """设置是否显示时间"""
        self.show_time = show_time

    def log(self, level, message):
        """写日志"""

        # 日志关闭
        if not self.enabled:
            return

        # 日志级别过滤
        if level < self.level:
            return

        level_name = self.LEVEL_NAMES.get(level, "UNKNOWN")

        if self.show_time:
            now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{now}] [{level_name}] {message}\n"
        else:
            log_line = f"[{level_name}] {message}\n"

        # 追加写入
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line)

    @property
    def is_debug_enabled(self):
        return self.enabled and self.LEVEL_DEBUG >= self.level

    @property
    def is_info_enabled(self):
        return self.enabled and self.LEVEL_INFO >= self.level

    def debug(self, message=""):
        if self.is_debug_enabled:
            self.log(self.LEVEL_DEBUG, message)

    def info(self, message=""):
        if self.is_info_enabled:
            self.log(self.LEVEL_INFO, message)

    def warning(self, message=""):
        if self.enabled and self.LEVEL_WARNING >= self.level:
            self.log(self.LEVEL_WARNING, message)

    def error(self, message=""):
        if self.enabled and self.LEVEL_ERROR >= self.level:
            self.log(self.LEVEL_ERROR, message)


class GSALogger:
    """支持多类型日志的日志管理器（性能日志、精度日志、日常信息日志）"""

    # 日志类型常量
    LOG_TYPE_PERFORMANCE = "performance"  # 性能日志
    LOG_TYPE_ACCURACY = "accuracy"  # 精度日志
    LOG_TYPE_NORMAL = "normal"  # 日常信息日志
    LOG_TYPE_ALL = "all"  # 所有日志

    # 日志级别
    LEVEL_DEBUG = 10
    LEVEL_INFO = 20
    LEVEL_WARNING = 30
    LEVEL_ERROR = 40

    LEVEL_NAMES = {
        LEVEL_DEBUG: "DEBUG",
        LEVEL_INFO: "INFO",
        LEVEL_WARNING: "WARNING",
        LEVEL_ERROR: "ERROR",
    }

    def __init__(
        self,
        log_dir="gsa_log/GLM/pyLog",
        dp_rank=0,
        tp_rank=0,
        log_type=LOG_TYPE_ALL,
        level=LEVEL_INFO,
        show_time=False,
    ):
        """
        :param log_dir: 日志目录
        :param dp_rank: 数据并行 rank
        :param tp_rank: 张量并行 rank
        :param log_type: 日志类型 (performance/accuracy/normal/all)
        :param level: 最低日志级别
        :param show_time: 是否打印时间
        """
        self.log_type = log_type
        self.level = level
        self.show_time = show_time

        # 基础文件名
        base_name = f"log_for_{dp_rank}_tp_{tp_rank}"

        # 根据日志类型创建不同的日志文件
        self.loggers = {}

        if log_type == GSALogger.LOG_TYPE_ALL:
            # 所有日志类型都打印，创建三个日志文件
            self.loggers[GSALogger.LOG_TYPE_PERFORMANCE] = self._create_logger(
                os.path.join(log_dir, f"{base_name}_performance.log")
            )
            self.loggers[GSALogger.LOG_TYPE_ACCURACY] = self._create_logger(
                os.path.join(log_dir, f"{base_name}_accuracy.log")
            )
            self.loggers[GSALogger.LOG_TYPE_NORMAL] = self._create_logger(
                os.path.join(log_dir, f"{base_name}_normal.log")
            )
        else:
            # 只创建指定类型的日志文件
            log_file = os.path.join(log_dir, f"{base_name}_{log_type}.log")
            self.loggers[log_type] = self._create_logger(log_file)

    def _create_logger(self, log_file):
        """创建单个日志文件的Logger实例"""
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        return Logger(
            log_file, enabled=True, level=self.level, show_time=self.show_time
        )

    def _get_level_name(self, level):
        """获取日志级别名称"""
        return self.LEVEL_NAMES.get(level, "UNKNOWN")

    def _format_message(self, message, level=None):
        """格式化日志消息"""
        level_name = self._get_level_name(level) if level else ""
        if self.show_time:
            now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            return f"[{now}] [{level_name}] {message}"
        else:
            return f"[{level_name}] {message}"

    def _write_log(self, log_type, level, message):
        """写日志到指定类型"""
        # 级别过滤
        if level < self.level:
            return

        if self.log_type == GSALogger.LOG_TYPE_ALL:
            # 所有模式：所有日志类型都写入对应文件
            for logger in self.loggers.values():
                logger.log(level, message)
        else:
            # 单类型模式：只写入对应文件
            if log_type in self.loggers:
                self.loggers[log_type].log(level, message)

    def performance_is_info(self):
        self.loggers[GSALogger.LOG_TYPE_PERFORMANCE].is_info_enabled

    def performance_is_debug(self):
        self.loggers[GSALogger.LOG_TYPE_PERFORMANCE].is_debug_enabled

    def accuracy_is_info(self):
        self.loggers[GSALogger.LOG_TYPE_ACCURACY].is_info_enabled

    def accuracy_is_debug(self):
        self.loggers[GSALogger.LOG_TYPE_ACCURACY].is_debug_enabled

    def normal_is_info(self):
        self.loggers[GSALogger.LOG_TYPE_NORMAL].is_info_enabled

    def normal_is_debug(self):
        self.loggers[GSALogger.LOG_TYPE_NORMAL].is_debug_enabled

    def performance_debug(self, message):
        """性能日志debug级别"""
        self._write_log(GSALogger.LOG_TYPE_PERFORMANCE, self.LEVEL_DEBUG, message)

    def performance_info(self, message):
        """性能日志info级别"""
        self._write_log(GSALogger.LOG_TYPE_PERFORMANCE, self.LEVEL_INFO, message)

    def performance_warning(self, message):
        """性能日志warning级别（错误日志）"""
        self._write_log(GSALogger.LOG_TYPE_PERFORMANCE, self.LEVEL_WARNING, message)

    def performance_error(self, message):
        """性能日志error级别"""
        self._write_log(GSALogger.LOG_TYPE_PERFORMANCE, self.LEVEL_ERROR, message)

    def accuracy_debug(self, message):
        """精度日志debug级别"""
        self._write_log(GSALogger.LOG_TYPE_ACCURACY, self.LEVEL_DEBUG, message)

    def accuracy_info(self, message):
        """精度日志info级别"""
        self._write_log(GSALogger.LOG_TYPE_ACCURACY, self.LEVEL_INFO, message)

    def accuracy_warning(self, message):
        """精度日志warning级别（错误日志）"""
        self._write_log(GSALogger.LOG_TYPE_ACCURACY, self.LEVEL_WARNING, message)

    def accuracy_error(self, message):
        """精度日志error级别"""
        self._write_log(GSALogger.LOG_TYPE_ACCURACY, self.LEVEL_ERROR, message)

    def normal_debug(self, message):
        """日常信息日志debug级别"""
        self._write_log(GSALogger.LOG_TYPE_NORMAL, self.LEVEL_DEBUG, message)

    def normal_info(self, message):
        """日常信息日志info级别"""
        self._write_log(GSALogger.LOG_TYPE_NORMAL, self.LEVEL_INFO, message)

    def normal_warning(self, message):
        """日常信息日志warning级别（错误日志）"""
        self._write_log(GSALogger.LOG_TYPE_NORMAL, self.LEVEL_WARNING, message)

    def normal_error(self, message):
        """日常信息日志error级别"""
        self._write_log(GSALogger.LOG_TYPE_NORMAL, self.LEVEL_ERROR, message)

    # 兼容原有接口
    def debug(self, message):
        """通用debug日志，写入normal类型"""
        self.normal_debug(message)

    def info(self, message):
        """通用info日志，写入normal类型"""
        self.normal_info(message)

    def warning(self, message):
        """通用warning日志，写入normal类型"""
        self.normal_warning(message)

    def error(self, message):
        """通用error日志，写入normal类型"""
        self.normal_error(message)

    @property
    def is_debug_enabled(self):
        return self.normal_is_info

    @property
    def is_info_enabled(self):
        return self.normal_is_debug
