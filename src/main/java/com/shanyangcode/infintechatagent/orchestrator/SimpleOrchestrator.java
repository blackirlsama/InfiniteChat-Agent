package com.shanyangcode.infintechatagent.orchestrator;

import com.shanyangcode.infintechatagent.agent.KnowledgeAgent;
import com.shanyangcode.infintechatagent.agent.ReasoningAgent;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;

/**
 * 简单协调器 - 协调多个Agent协同工作
 */
@Service
@Slf4j
public class SimpleOrchestrator {

    @Resource
    private KnowledgeAgent knowledgeAgent;

    @Resource
    private ReasoningAgent reasoningAgent;

    // 知识查询关键词
    private static final List<String> KNOWLEDGE_KEYWORDS = Arrays.asList(
            "查询", "了解", "什么是", "介绍", "解释", "说明"
    );

    /**
     * 处理用户请求
     */
    public String process(Long sessionId, String userInput) {
        log.info("[SimpleOrchestrator] 开始处理请求，sessionId: {}", sessionId);

        String enhancedInput = userInput;

        // 判断是否需要知识检索
        if (needKnowledgeRetrieval(userInput)) {
            log.info("[SimpleOrchestrator] 检测到知识查询需求，调用KnowledgeAgent");
            String knowledgeResult = knowledgeAgent.execute(sessionId, userInput);

            if (knowledgeResult != null && !knowledgeResult.trim().isEmpty()) {
                enhancedInput = "参考知识：\n" + knowledgeResult + "\n\n用户问题：" + userInput;
                log.info("[SimpleOrchestrator] 知识检索成功，增强输入");
            } else {
                log.info("[SimpleOrchestrator] 知识检索无结果，使用原始输入");
            }
        } else {
            log.info("[SimpleOrchestrator] 无需知识检索，直接推理");
        }

        // 调用推理Agent生成最终回复
        String finalResult = reasoningAgent.execute(sessionId, enhancedInput);
        log.info("[SimpleOrchestrator] 处理完成");

        return finalResult;
    }

    /**
     * 判断是否需要知识检索
     */
    private boolean needKnowledgeRetrieval(String input) {
        return KNOWLEDGE_KEYWORDS.stream().anyMatch(input::contains);
    }
}
