input的格式：

单key格式:
optype ns key offset length value

batch格式:
optype batch ns=x 或者是 ns x （ns 必填，不可省略，不能为空）
key offset length value
key offset length value
/batch

其中value可省略：
value是固定字段（op ns key offset length）之后的剩余全部内容，允许包含空格。
当testmode=1时，由系统生成，用户写了也不生效。 生成规则：key 转成128位整数，选低64位，拼接高16位，拼接offset，拼接length，拼接当前时间（%Y-%m-%d %H:%M:%S），中间用空格隔开。 种子串只写一次不补齐（超length截断），后面是'A'或上次残留；同一个batch共用同一时间戳
当testmode=2时，由value文件中的value指定。   程序启动时，缓冲区会全部初始化为A，每次只写value前缀，不补齐，后面是啥就是啥。




optype: get put exist delete

ns 是请求级 namespace 名（字符串，可与数字形式的名字）：经 IoOptions.kvns 传给 kv client，由 kv client 哈希派生协议帧 namespace id（XXH32，种子 5381）——须与对端存储的 namespace 名一致（M800 为 Ceph pool 名），同一名字对端即同一 namespace。**ns 必填且不能为空**（单 key 行必须四字段、batch 头必带 ns）：空串也会哈希派生出一个独立的 namespace id（即"空名字的 namespace"），并非对端缺省 namespace；缺 ns 或空 ns 的输入在程序启动时报错退出（fail-fast），须写对端实际 namespace 名。

命令与命令之间可插入sleep x 模拟脉冲。所有线程同步暂停X秒。

其他规则：
单key格式必须四字段（ns key offset length），缺 ns（仅三字段 key offset length）启动报错退出
字段须满足 kv 协议约束（协议层 BatchStore/BatchRetrieve 的校验规则）：key 低 64 位不能为全 0（key=0 非法）、offset 与 length 均须 512 字节对齐、length 不能为 0、length 上限 0xFFFFFF（24 位）。违反任一约束在程序启动时报错退出（fail-fast），不会等到运行期以任务失败暴露
对文件逐行读取，去掉尾部\n，如果最后一个字符是\r去掉，行首尾trim，trim字符集是空格和\t
trim后为空，跳过该行。trim后以sleep开头，按sleep处理
提取行首操作符，提取失败则启动报错退出
如果以batch开头，开始batch。如果完全等于/batch batch结束。 文件结束时batch未闭合（无/batch）：启动报错退出，提示管理员补全
否则按照单行请求解析：必须四字段（ns key offset length，ns 非空）。失败则启动报错退出
任何非法输入数据都不会跳过或静默继续，直接返回启动错误，由管理员修正后重新启动
不支持注释

操作符只认上面几个，大小写敏感

请求明细日志：目录与文件名在 test_config.conf 的 log 参数段配置。get 任务明细写入 log.get_dir 目录下 log.get_file_name 文件（缺省 log/request/get_*.log），每个entry一行，形如 (get), ns (tenant), return batch code (0,0,259), key code (0), inputkey (原文key), key+offset (原文key 高16位 offset，三数空格分隔), offset (块内偏移), len (=key位数+offset位数，纯显示用途), tail (尾8字节), value (读回缓冲实际内容，置于行末——testmode=2为完整块，长块不挡前序短字段)，字段间", "分隔；return batch code为整批全部entry状态码逗号拼接（每行重复整批码），key code为该entry自身返回码；get无论entry成败都读回缓冲实际内容——IO前缓冲预填哨兵串unknownvaluebeforegetrequest,,,（未命中时value显示哨兵串即为"没读到数据"标记）；value读回长度testmode=1为key低64位位数+高16位位数+offset位数+2（恰覆盖种子串"keyLow keyHigh offset"段），testmode=2为完整块。put/exist/delete任务明细写入 log.other_dir 目录下 log.other_file_name 文件（缺省 log/request/other_*.log）：put每个entry一行、格式与get同款（opcode位置为put；value仅成功entry读回，失败以-占位）；exist与delete每任务一行聚合格式，形如 (exist), ns (tenant), all_block_ids[key1,key2] key_high[原文key 高16位,原文key 高16位] prefix_hit_len (前缀命中key数)，delete无前缀命中信息、prefix_hit_len以-占位。perf=off全量记录；perf=on时全部操作仅打印失败请求明细（成功请求不打印：get/put为任务级或该entry返回码非0含未命中，exist为任一key未存在，delete为任一key未命中或失败），明细日志文件仍创建，主日志仍按log.level输出
