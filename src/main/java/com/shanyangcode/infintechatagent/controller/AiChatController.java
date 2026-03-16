package com.shanyangcode.infintechatagent.controller;


import com.shanyangcode.infintechatagent.Monitor.MonitorContext;
import com.shanyangcode.infintechatagent.Monitor.MonitorContextHolder;
import com.shanyangcode.infintechatagent.ai.AiChat;
import com.shanyangcode.infintechatagent.model.dto.ChatRequest;
import com.shanyangcode.infintechatagent.model.dto.KnowledgeRequest;
import com.shanyangcode.infintechatagent.orchestrator.SimpleOrchestrator;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/**
 * AI聊天控制器
 * 提供聊天、流式聊天和知识插入等功能
 */
@RestController
@Slf4j
public class AiChatController {

    @Resource
    private AiChat aiChat; // AI聊天服务接口

    @Resource
    private SimpleOrchestrator simpleOrchestrator; // 多Agent协调器

    @Resource
    private EmbeddingStoreIngestor embeddingStoreIngestor; // 向量数据库导入服务

    @Value("${rag.docs-path}")
    private String docsPath; // 文档存储路径配置

    private final  String  TARGET_FILENAME = "InfiniteChat.md"; // 目标文件名常量

    /**
     * 聊天接口（已注释）
     * @param sessionId 会话ID
     * @param prompt 用户输入
     * @return 聊天回复
     */
//    @GetMapping("/chat")
//    public String chat(String sessionId, String prompt) {
//        return aiChat.chat(sessionId, prompt);
//    }

    /**
     * POST方式聊天接口
     * @param chatRequest 包含会话ID和用户输入的请求对象
     * @return 聊天回复
     */
    @PostMapping("/chat")
    public String chat(@RequestBody ChatRequest chatRequest) {
        // 设置监控上下文
        MonitorContextHolder.setContext(MonitorContext.builder().userId(chatRequest.getUserId()).sessionId(chatRequest.getSessionId()).build());
        // 执行聊天并获取结果
        String chat = aiChat.chat(chatRequest.getSessionId(), chatRequest.getPrompt());
        // 清除监控上下文
        MonitorContextHolder.clearContext();
        return chat;
    }


    /**
     * 流式聊天接口（已注释）
     * @param chatRequest 包含会话ID和用户输入的请求对象
     * @return 流式字符串响应
     */
//    @PostMapping("/streamChat")
//    public Flux<String> streamChat(@RequestBody ChatRequest chatRequest) {
//        return aiChat.streamChat(chatRequest.getSessionId(), chatRequest.getPrompt());
//    }




    /**
     * POST方式流式聊天接口
     * @param chatRequest 包含会话ID和用户输入的请求对象
     * @return 流式字符串响应
     */
    @PostMapping("/streamChat")
    public Flux<String> streamChat(@RequestBody ChatRequest chatRequest) {
        // 构建监控上下文
        MonitorContext context = MonitorContext.builder()
                .userId(chatRequest.getUserId())
                .sessionId(chatRequest.getSessionId())
                .build();

        // 使用defer延迟执行，确保在订阅时才设置上下文
        return Flux.defer(() -> {
            // 设置监控上下文
            MonitorContextHolder.setContext(context);
            // 执行流式聊天，并在完成时清除上下文
            return aiChat.streamChat(chatRequest.getSessionId(), chatRequest.getPrompt())
                    .doFinally(signal -> MonitorContextHolder.clearContext());
        });
    }


    /**
     * 多Agent协同聊天接口（新增）
     * @param chatRequest 包含会话ID和用户输入的请求对象
     * @return 聊天回复
     */
    @PostMapping("/multiAgentChat")
    public String multiAgentChat(@RequestBody ChatRequest chatRequest) {
        // 设置监控上下文
        MonitorContextHolder.setContext(MonitorContext.builder()
                .userId(chatRequest.getUserId())
                .sessionId(chatRequest.getSessionId())
                .build());

        // 调用多Agent协调器处理
        String result = simpleOrchestrator.process(chatRequest.getSessionId(), chatRequest.getPrompt());

        // 清除监控上下文
        MonitorContextHolder.clearContext();
        return result;
    }


    /**
     * 插入知识接口
     * @param knowledgeRequest 包含问题和答案的知识请求对象
     * @return 操作结果字符串
     */
    @PostMapping("/insert")
    public String insertKnowledge(@RequestBody KnowledgeRequest knowledgeRequest) {
        // 1. 格式化内容为Q&A格式
        String formattedContent = String.format("### Q：%s\n\nA：%s", knowledgeRequest.getQuestion(), knowledgeRequest.getAnswer());

        // 2. 写入物理文件 (InfiniteChat.md)
        boolean writeSuccess = appendToFile(formattedContent, knowledgeRequest.getSourceName());
        if (!writeSuccess) {
            return "插入失败：无法写入本地文件";
        }

        // 3. 存入向量数据库 (RAG)
        try {
            // 设置来源元数据
            String sourceName = (knowledgeRequest.getSourceName() != null) ? knowledgeRequest.getSourceName() : TARGET_FILENAME;
            Metadata metadata = Metadata.from("file_name", sourceName);

            // 创建文档并 Embedding
            Document document = Document.from(formattedContent, metadata);
            embeddingStoreIngestor.ingest(document);

            log.info("RAG - 新增知识点成功: {}", knowledgeRequest.getQuestion());
            return "插入成功：已同步至 " + knowledgeRequest.getSourceName() + " 及向量数据库";
        } catch (Exception e) {
            log.error("RAG - 向量化失败", e);
            return "插入部分成功：文件已写入，但向量库更新失败";
        }
    }



    /**
     * 向文件追加内容（线程安全）
     * @param content 要追加的内容
     * @param sourceName 源文件名
     * @return 是否写入成功
     */
    private synchronized boolean appendToFile(String content, String sourceName) {
        try {
            // 拼接完整路径
            Path filePath = Paths.get(docsPath, sourceName);
            log.info("文件实际写入位置: {}", filePath.toAbsolutePath());
            // 如果文件不存在，先创建
            if (!Files.exists(filePath)) {
                Files.createDirectories(filePath.getParent());
                Files.createFile(filePath);
            }

            // 准备要写入的文本，前后加换行符确保格式独立
            String textToAppend = "\n\n" + content;

            // 执行追加写入
            Files.writeString(
                    filePath,
                    textToAppend,
                    StandardOpenOption.APPEND,
                    StandardOpenOption.CREATE
            );
            return true;
        } catch (IOException e) {
            log.error("RAG - 写入本地文件失败: {}", e.getMessage(), e);
            return false;
        }
    }
}



