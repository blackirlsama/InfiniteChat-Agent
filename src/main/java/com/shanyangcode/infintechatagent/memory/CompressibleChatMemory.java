package com.shanyangcode.infintechatagent.memory;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.store.memory.chat.ChatMemoryStore;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.SessionCallback;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

@Slf4j
public class CompressibleChatMemory implements ChatMemory {

    private final Object id;
    private final ChatMemoryStore store;
    private final TokenCountChatMemoryCompressor compressor;
    private final int maxMessages;
    private final int tokenThreshold;
    private final int fallbackRecentRounds;
    private final RedisTemplate<String, Object> redisTemplate;
    private final int lockExpireSeconds;
    private final int lockRetryTimes;
    private final int lockRetryIntervalMs;

    public CompressibleChatMemory(Object id,
                                  ChatMemoryStore store,
                                  TokenCountChatMemoryCompressor compressor,
                                  int maxMessages,
                                  int tokenThreshold,
                                  int fallbackRecentRounds,
                                  RedisTemplate<String, Object> redisTemplate,
                                  int lockExpireSeconds,
                                  int lockRetryTimes,
                                  int lockRetryIntervalMs) {
        this.id = id;
        this.store = store;
        this.compressor = compressor;
        this.maxMessages = maxMessages;
        this.tokenThreshold = tokenThreshold;
        this.fallbackRecentRounds = fallbackRecentRounds;
        this.redisTemplate = redisTemplate;
        this.lockExpireSeconds = lockExpireSeconds;
        this.lockRetryTimes = lockRetryTimes;
        this.lockRetryIntervalMs = lockRetryIntervalMs;
    }

    @Override
    public Object id() {
        return id;
    }

    @Override
    public void add(ChatMessage message) {
        String lockKey = "chat:memory:lock:" + id;
        String lockValue = UUID.randomUUID().toString();
        boolean lockAcquired = false;
        int retryCount = 0;

        try {
            while (retryCount < lockRetryTimes && !lockAcquired) {
                lockAcquired = acquireLock(lockKey, lockValue);
                if (!lockAcquired) {
                    retryCount++;
                    Thread.sleep(lockRetryIntervalMs);
                    log.debug("会话{}获取锁失败，重试{}/{}", id, retryCount, lockRetryTimes);
                }
            }

            if (!lockAcquired) {
                log.warn("会话{}获取锁失败，降级处理", id);
                List<ChatMessage> messages = new ArrayList<>(store.getMessages(id));
                messages.add(message);
                store.updateMessages(id, messages);
                return;
            }

            log.debug("会话{}获取锁成功", id);
            List<ChatMessage> messages = new ArrayList<>(store.getMessages(id));
            messages.add(message);

            boolean needCompress = false;
            int currentTokenCount = 0;
            if (compressor != null) {
                currentTokenCount = compressor.estimateTokens(messages);
                needCompress = messages.size() > maxMessages || currentTokenCount > tokenThreshold;
            } else {
                needCompress = messages.size() > maxMessages;
            }

            if (needCompress) {
                log.info("🔄 会话{}触发压缩 - 消息数:{}/{}, Token数:{}/{}",
                    id, messages.size(), maxMessages, currentTokenCount, tokenThreshold);
                try {
                    messages = compressor.compress(messages);
                    log.info("✅ 会话{}压缩完成 - 压缩后消息数:{}, Token数:{}",
                        id, messages.size(), compressor.estimateTokens(messages));
                } catch (Exception e) {
                    log.error("❌ 会话{}压缩失败，降级保留最近{}轮", id, fallbackRecentRounds, e);
                    if (messages.size() > fallbackRecentRounds) {
                        messages = messages.subList(messages.size() - fallbackRecentRounds, messages.size());
                    }
                }
                atomicUpdateMessages(messages);
            } else {
                log.debug("会话{}无需压缩 - 消息数:{}, Token数:{}", id, messages.size(), currentTokenCount);
                store.updateMessages(id, messages);
            }
        } catch (InterruptedException e) {
            log.error("会话{}处理中断", id, e);
            Thread.currentThread().interrupt();
        } finally {
            if (lockAcquired) {
                releaseLock(lockKey, lockValue);
                log.debug("会话{}释放锁", id);
            }
        }
    }

    @Override
    public List<ChatMessage> messages() {
        return store.getMessages(id);
    }

    @Override
    public void clear() {
        String lockKey = "chat:memory:lock:" + id;
        String lockValue = UUID.randomUUID().toString();
        if (acquireLock(lockKey, lockValue)) {
            try {
                store.deleteMessages(id);
            } finally {
                releaseLock(lockKey, lockValue);
            }
        } else {
            log.error("清理会话{}记忆时获取锁失败", id);
        }
    }

    // 新增：Redis原子更新消息（事务方式）
    private void atomicUpdateMessages(List<ChatMessage> messages) {
        try {
            redisTemplate.execute(new SessionCallback<Object>() {
                @Override
                public Object execute(org.springframework.data.redis.core.RedisOperations operations) {
                    operations.multi();
                    store.deleteMessages(id);
                    store.updateMessages(id, messages);
                    return operations.exec();
                }
            });
            log.debug("会话{}消息原子更新完成", id);
        } catch (Exception e) {
            log.error("会话消息原子更新失败", id, e);
            store.updateMessages(id, messages);
        }
    }

    // 新增：获取Redis分布式锁
    private boolean acquireLock(String lockKey, String lockValue) {
        return Boolean.TRUE.equals(redisTemplate.opsForValue().setIfAbsent(
                lockKey, lockValue, lockExpireSeconds, java.util.concurrent.TimeUnit.SECONDS));
    }

    // 新增：释放Redis分布式锁（Lua脚本保证原子性）
    private void releaseLock(String lockKey, String lockValue) {
        String script = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end";
        try {
            redisTemplate.execute(
                    new org.springframework.data.redis.core.script.DefaultRedisScript<>(script, Long.class),
                    Collections.singletonList(lockKey),
                    lockValue
            );
        } catch (Exception e) {
            log.error("释放会话{}的分布式锁失败", id, e);
        }
    }
}