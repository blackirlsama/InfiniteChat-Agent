package com.shanyangcode.infintechatagent.controller;


import com.shanyangcode.infintechatagent.ai.AiChat;
import jakarta.annotation.Resource;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class AiChatController {

    @Resource
    private AiChat aiChat;

    @GetMapping("/chat")
    public String chat(String sessionId, String prompt) {
        return aiChat.chat(sessionId, prompt);
    }
}



