package com.shanyangcode.infintechatagent.agent;

import com.shanyangcode.infintechatagent.tool.RagTool;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 知识Agent - 负责知识检索
 */
@Component
@Slf4j
public class KnowledgeAgent implements Agent {

    @Resource
    private RagTool ragTool;

    @Override
    public String execute(Long sessionId, String input) {
        log.info("[KnowledgeAgent] 执行知识检索，sessionId: {}, input: {}", sessionId, input);
        try {
            String result = ragTool.retrieve(input);
            log.info("[KnowledgeAgent] 检索结果: {}", result != null ? result.substring(0, Math.min(100, result.length())) : "null");
            return result != null ? result : "";
        } catch (Exception e) {
            log.error("[KnowledgeAgent] 检索失败", e);
            return "";
        }
    }

    @Override
    public String getAgentName() {
        return "KnowledgeAgent";
    }
}
