input的格式：

单key格式:
optype nsid key offset length value

batch格式:
optype batch nsid=x 或者是 nsid x （nsid可省略，缺省为0）
key offset length value
key offset length value
/batch

其中value可省略：
value是固定字段（op nsid key offset length）之后的剩余全部内容，允许包含空格。
当testmode=1时，由系统生成，用户写了也不生效。 生成规则：key 转成128位整数，选低64位，拼接高16位，拼接offset，拼接length，拼接当前时间（%Y-%m-%d %H:%M:%S），中间用空格隔开。 种子串只写一次不补齐（超length截断），后面是'A'或上次残留；同一个batch共用同一时间戳
当testmode=2时，由value文件中的value指定。   程序启动时，缓冲区会全部初始化为A，每次只写value前缀，不补齐，后面是啥就是啥。




optype: get put exist delete

命令与命令之间可插入sleep x 模拟脉冲。所有线程同步暂停X秒。

其他规则：
单key格式中，如果只有3个整数，那么是nsid缺省，默认为0
字段须满足 kv 协议约束（协议层 BatchStore/BatchRetrieve 的校验规则）：key 低 64 位不能为全 0（key=0 非法）、offset 与 length 均须 512 字节对齐、length 不能为 0、length 上限 0xFFFFFF（24 位）。违反任一约束在程序启动时报错退出（fail-fast），不会等到运行期以任务失败暴露
对文件逐行读取，去掉尾部\n，如果最后一个字符是\r去掉，行首尾trim，trim字符集是空格和\t
trim后为空，跳过该行。trim后以sleep开头，按sleep处理
提取行首操作符，提取失败则启动报错退出
如果以batch开头，开始batch。如果完全等于/batch batch结束。 文件结束时batch未闭合（无/batch）：启动报错退出，提示管理员补全
否则按照单行请求解析：先试四整数，再试三整数。都失败则启动报错退出
任何非法输入数据都不会跳过或静默继续，直接返回启动错误，由管理员修正后重新启动
不支持注释

操作符只认上面几个，大小写敏感

请求明细日志：put/get/exist/delete任务的响应明细写入readlogdir目录下req_*.log（两种模式共用一个文件，不拆分），每个entry一行，opcode字段区分操作类型；行内batchCode为任务级返回码，其后依次为input原文key、key_low（key低64位十进制）、key_high（key高16位十进制）、offset、len（=key_low位数+offset位数，纯显示用途）；put/get行另有entryCode（该entry自身返回码）与value（读回缓冲实际内容：testmode=1读生成前缀部分、testmode=2读完整块）及tail（尾8字节）；exist行无entryCode，含exists标志与prefix_hit_len（前缀命中key数）；delete行只有entryCode与key分量（无value载荷）。perf=off全量记录；perf=on只记录有错误码的行（put/get/delete按entryCode、exist按batchCode过滤），且全量主日志不产生任何info级别日志
