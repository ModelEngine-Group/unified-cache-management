input的格式：

单key格式:
optype nsid key offset length value

batch格式:
optype batch nsid=x 或者是 nsid x
key offset length value
key offset length value
/batch

其中value可省略：
当testmode=1时，由系统生成，用户写了也不生效。 生成规则：key 转成128位整数，选低64位，拼接高16位，拼接offset，拼接length，拼接当前时间（%Y-%m-%d %H:%M:%S），中间用空格隔开
当testmode=2时，由value文件中的value指定。   程序启动时，缓冲区会全部初始化为A，每次只修改value大小的长度，后面是啥就是啥。



optype: get put exists delete 

命令与命令之间可插入sleep x 模拟脉冲。所有线程同步暂停X秒。

其他规则：
单key格式中，如果只有3个整数，那么是nsid缺省，默认为0
对文件逐行读取，去掉尾部\n，如果最后一个字符是\r去掉，行首尾trim，trim字符集是空格和\t
trim后为空，跳过该行。trim后以sleep开头，按sleep处理
提取行首操作符，提取失败则警告，跳过该行
如果以batch开头，开始batch。如果完全等于/batch batch结束
否则按照单行请求解析：先试四整数，再试三整数。都失败则警告并跳过
不支持注释

操作符只认上面几个，大小写敏感
