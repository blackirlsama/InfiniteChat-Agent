package com.shanyangcode.infintechatagent.config;

import com.shanyangcode.infintechatagent.memory.TokenCountChatMemoryCompressor;
import jakarta.annotation.PostConstruct;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "chat.memory")
@Data
@Slf4j
public class ChatMemoryConfig {

    private int maxMessages = 20;
    private Compression compression = new Compression();
    private Redis redis = new Redis();

    @PostConstruct
    public void init() {
        log.info("=== 会话记忆压缩配置已加载 ===");
        log.info("最大消息数: {}", maxMessages);
        log.info("Token阈值: {}", compression.getTokenThreshold());
        log.info("保留最近轮数: {}", compression.getRecentRounds());
        log.info("最近对话Token上限: {}", compression.getRecentTokenLimit());
        log.info("降级保留轮数: {}", compression.getFallbackRecentRounds());
        log.info("分布式锁过期时间: {}秒", redis.getLock().getExpireSeconds());
        log.info("分布式锁重试次数: {}", redis.getLock().getRetryTimes());
        log.info("==============================");
    }

    @Data
    public static class Compression {
        private int tokenThreshold = 6000;
        private int recentRounds = 5;
        private int recentTokenLimit = 2000;
        private int summaryTokenLimit = 500;
        private String summaryPrompt;
        private int fallbackRecentRounds = 10;
    }

    @Data
    public static class Redis {
        private int ttlSeconds = 3600;
        private Lock lock = new Lock();
    }

    @Data
    public static class Lock {
        private int expireSeconds = 5;
        private int retryTimes = 3;
        private int retryIntervalMs = 100;
    }

    @Bean
    public TokenCountChatMemoryCompressor tokenCountChatMemoryCompressor() {
        log.info("创建TokenCountChatMemoryCompressor Bean");
        return new TokenCountChatMemoryCompressor(
            compression.getRecentRounds(),
            compression.getRecentTokenLimit()
        );
    }
}
