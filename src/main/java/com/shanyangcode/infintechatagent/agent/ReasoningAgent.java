package com.shanyangcode.infintechatagent.agent;

import com.shanyangcode.infintechatagent.ai.AiChat;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 推理Agent - 负责对话和推理
 */
@Component
@Slf4j
public class ReasoningAgent implements Agent {

    @Resource
    private AiChat aiChat;

    @Override
    public String execute(Long sessionId, String input) {
        log.info("[ReasoningAgent] 执行推理，sessionId: {}, input: {}", sessionId, input);
        String result = aiChat.chat(sessionId, input);
        log.info("[ReasoningAgent] 推理完成");
        return result;
    }

    @Override
    public String getAgentName() {
        return "ReasoningAgent";
    }
}
