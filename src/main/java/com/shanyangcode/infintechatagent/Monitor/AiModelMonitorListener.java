package com.shanyangcode.infintechatagent.Monitor;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;

import dev.langchain4j.model.chat.listener.ChatModelErrorContext;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.listener.ChatModelRequestContext;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;
import dev.langchain4j.model.output.TokenUsage;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;




@Component
@Slf4j
public class AiModelMonitorListener implements ChatModelListener {

    // 定义一个 Key，用于在请求和响应之间传递开始时间
    private static final String START_TIME_KEY = "request_start_time";

    private static final String MONITOR_CONTEXT_KEY = "monitor_context";

    @Resource
    private AiModelMetricsCollector aiModelMetricsCollector;


    @Override
    public void onRequest(ChatModelRequestContext requestContext) {

        requestContext.attributes().put(START_TIME_KEY, Instant.now());
        // 从监控上下文中获取信息
        MonitorContext context = MonitorContextHolder.getContext();

        if (context == null) {
            // 记录错误日志
            log.error("MonitorContext is null when processing request");
            return;
        }
        String userId = context.getUserId() != null ? context.getUserId().toString() : "unknown";
        String sessionId = context.getSessionId() != null ? context.getSessionId().toString() : "unknown";
        requestContext.attributes().put(MONITOR_CONTEXT_KEY, context);
        // 获取模型名称
        String modelName = requestContext.chatRequest().modelName();

        log.info(">>> AI请求开始 | 用户: {} | 会话: {} | 模型: {}", userId, sessionId, modelName);
        // 记录请求指标
        aiModelMetricsCollector.recordRequest(userId, sessionId, modelName, "started");
    }

    @Override
    public void onResponse(ChatModelResponseContext responseContext) {
        String modelName = responseContext.chatResponse().metadata().modelName();
        // 从属性中获取监控信息（由 onRequest 方法存储）
        Map<Object, Object> attributes = responseContext.attributes();
        // 1. 从监控上下文中获取信息
        MonitorContext context = (MonitorContext) attributes.get(MONITOR_CONTEXT_KEY);

        if (context == null) {
            log.warn("监控上下文丢失，无法记录响应指标 - Model: {}", responseContext.chatResponse().modelName());
            return;
        }
        
        String userId = context.getUserId().toString();
        String sessionId = context.getSessionId().toString();
        // 2. 计算耗时
        Duration durationMs = calculateDuration(attributes);

        // 3. 获取 Token 使用情况
        TokenUsage tokenUsage = responseContext.chatResponse().metadata().tokenUsage();

        // 4. 打印格式化日志
        log.info("<<< AI请求成功 | 用户: {} | 会话: {} | 模型: {} | 耗时: {}ms | Tokens: [In:{}, Out:{}, Total:{}]", userId, sessionId, modelName, durationMs.toMillis(), tokenUsage != null ? tokenUsage.inputTokenCount() : 0, tokenUsage != null ? tokenUsage.outputTokenCount() : 0, tokenUsage != null ? tokenUsage.totalTokenCount() : 0);
        aiModelMetricsCollector.recordRequest(userId, sessionId, modelName, "success");
        aiModelMetricsCollector.recordResponseTime(userId, sessionId, modelName, durationMs);

        if (tokenUsage != null) {
            aiModelMetricsCollector.recordTokenUsage(userId, sessionId, modelName, "input", tokenUsage.inputTokenCount());
            aiModelMetricsCollector.recordTokenUsage(userId, sessionId, modelName, "output", tokenUsage.outputTokenCount());
            aiModelMetricsCollector.recordTokenUsage(userId, sessionId, modelName, "total", tokenUsage.totalTokenCount());
        }
    }

    @Override
    public void onError(ChatModelErrorContext errorContext) {
        MonitorContext context = MonitorContextHolder.getContext();

        Map<Object, Object> attributes = errorContext.attributes();
        Duration durationMs = calculateDuration(attributes);

        if (context == null) {
            // 尝试从 attributes 补救
            context = (MonitorContext) errorContext.attributes().get(MONITOR_CONTEXT_KEY);
        }

        if (context == null) {
            log.warn("监控上下文丢失，无法记录错误指标 - Error: {}", errorContext.error().getMessage());
            return;
        }
        
        String userId = context.getUserId().toString();
        String sessionId = context.getSessionId().toString();
        String modelName = errorContext.chatRequest().modelName();
        String errorMessage = errorContext.error().getMessage();
        log.error("AI 请求失败 | 耗时: {}ms | 错误原因: {}", durationMs.toMillis(), errorContext.error().getMessage());

        // 记录失败请求
        aiModelMetricsCollector.recordRequest(userId, sessionId, modelName, "error");
        aiModelMetricsCollector.recordError(userId, sessionId, modelName, errorMessage);
        aiModelMetricsCollector.recordResponseTime(userId, sessionId, modelName, durationMs);
    }

    /**
     * 从 attributes 中取出开始时间并计算间隔
     */
    private Duration calculateDuration(Map<Object, Object> attributes) {
        Instant startTime = (Instant) attributes.get(START_TIME_KEY);
        if (startTime != null) {
            return Duration.between(startTime, Instant.now());
        }
        return Duration.ZERO;
    }
}