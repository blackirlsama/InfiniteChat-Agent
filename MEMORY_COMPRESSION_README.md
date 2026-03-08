# 分布式会话记忆压缩策略实现文档

## 一、核心实现原理

### 1.1 压缩策略
- **滑动窗口**：保留最近5轮完整对话（约2000 tokens）
- **历史摘要**：更早的对话压缩为简短摘要（提取前3条消息的前50字符）
- **自动触发**：消息数超过20条时自动压缩

### 1.2 工作流程
```
消息总数 > maxMessages (20条)
    ↓
分割消息：前N轮 + 最近5轮
    ↓
生成历史摘要（简单文本截取）
    ↓
组装：[摘要消息] + [最近5轮完整对话]
    ↓
存储到Redis
```

## 二、核心类说明

### 2.1 TokenCountChatMemoryCompressor
**位置**: `com.shanyangcode.infintechatagent.memory.TokenCountChatMemoryCompressor`

**职责**: 执行消息压缩逻辑

**关键方法**:
- `compress(List<ChatMessage>)`: 压缩消息列表
- `generateSummary(List<ChatMessage>)`: 生成简单文本摘要
- `estimateTokens(List<ChatMessage>)`: 估算token数量（字符数/4）

### 2.2 CompressibleChatMemory
**位置**: `com.shanyangcode.infintechatagent.memory.CompressibleChatMemory`

**职责**: 实现ChatMemory接口，集成压缩器与Redis存储

### 2.3 ChatMemoryConfig
**位置**: `com.shanyangcode.infintechatagent.config.ChatMemoryConfig`

**职责**: 配置压缩参数并注册Bean

## 三、配置说明

### 3.1 application.yml 配置
```yaml
chat:
  memory:
    compression:
      recent-rounds: 5          # 保留最近N轮对话
      recent-token-limit: 2000  # 最近对话token上限
```

### 3.2 参数调优建议
| 参数 | 默认值 | 调优建议 |
|------|--------|----------|
| recent-rounds | 5 | 根据业务场景调整，客服场景建议3-5轮 |
| recent-token-limit | 2000 | 监控实际token使用，预留20%缓冲 |

## 四、使用示例

### 4.1 自动压缩
无需额外代码，框架自动触发：
```java
aiChat.chat(memoryId, "用户消息"); // 超过20条自动压缩
```

## 五、监控与调试

### 5.1 日志配置
```yaml
logging:
  level:
    com.shanyangcode.infintechatagent.memory: DEBUG
```

### 5.2 关键日志
- `Recent messages exceed token limit`: 最近对话超限警告
- `Compressed X messages to Y messages`: 压缩完成信息

## 六、注意事项

1. **简化实现**: 当前版本使用文本截取生成摘要，不调用LLM（避免API成本）
2. **Redis存储**: 确保Redis配置的TTL足够长（当前3600秒）
3. **并发安全**: 当前实现未加锁，高并发场景需考虑分布式锁

## 七、后续优化方向

1. **LLM摘要**: 集成LLM生成智能摘要（需解决API调用问题）
2. **异步压缩**: 将压缩操作改为异步执行
3. **缓存摘要**: 对相同历史消息的摘要结果进行缓存
