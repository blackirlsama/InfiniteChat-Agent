package com.shanyangcode.infintechatagent.tool;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.stream.Collectors;

@Component
@Slf4j
public class RagTool {

    @Resource
    private EmbeddingStoreIngestor embeddingStoreIngestor;

    // 新增依赖：向量存储（需提前配置，如Milvus/Chroma/Elasticsearch等）
    @Resource
    private EmbeddingStore<TextSegment> embeddingStore;

    // 新增依赖：Embedding模型（将文本转为向量，如OpenAI/智谱AI/本地模型）
    @Resource
    private EmbeddingModel embeddingModel;

    @Value("${rag.docs-path}")
    private String docsPath;

    // 可配置检索返回条数
    @Value("${rag.retrieve-top-k:3}")
    private Integer retrieveTopK;

    /**
     * 【适配 langchain4j 1.1.0 正式版】知识检索方法
     * @param query 检索关键词/问题
     * @return 拼接后的检索结果
     */
    public String retrieve(String query) {
        if (query == null || query.isBlank()) {
            log.warn("[RagTool] 检索关键词为空，返回空结果");
            return "";
        }

        try {
            // 1. 文本转向量
            Embedding queryEmbedding = embeddingModel.embed(query).content();

            // 2. 使用新版API - EmbeddingSearchRequest
            EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                    .queryEmbedding(queryEmbedding)
                    .maxResults(retrieveTopK)
                    .minScore(0.0)
                    .build();

            EmbeddingSearchResult<TextSegment> searchResult = embeddingStore.search(searchRequest);
            List<EmbeddingMatch<TextSegment>> matches = searchResult.matches();

            if (matches.isEmpty()) {
                log.info("[RagTool] 未检索到相关知识，query: {}", query);
                return "";
            }

            // 3. 拼接结果（兼容空值，避免空指针）
            String result = matches.stream()
                    .map(match -> {
                        // 安全获取元数据和文本 - 修复getString参数问题
                        String fileName = "未知文件";
                        if (match.embedded() != null && match.embedded().metadata() != null) {
                            // 1.1.0版本：先获取值，为空则用默认值
                            String fileNameFromMeta = match.embedded().metadata().getString("file_name");
                            if (fileNameFromMeta != null && !fileNameFromMeta.isBlank()) {
                                fileName = fileNameFromMeta;
                            }
                        }
                        String text = match.embedded() != null ? match.embedded().text() : "无内容";
                        double score = match.score() != null ? match.score() : 0.0;

                        return String.format("【来源：%s | 相似度：%.2f】\n%s",
                                fileName, score, text);
                    })
                    .collect(Collectors.joining("\n\n---\n\n"));

            log.info("[RagTool] 检索成功，匹配到 {} 条结果，query: {}", matches.size(), query);
            return result;

        } catch (Exception e) {
            log.error("[RagTool] 知识检索失败，query: {}", query, e);
            return "";
        }
    }

    /**
     * 原有方法：新增知识点到RAG
     */
    @dev.langchain4j.agent.tool.Tool("当用户想要保存问答对、知识点或者向知识库添加新信息时调用此工具。将问题、答案和目标文件名作为参数。")
    public String addKnowledgeToRag(String question, String answer, String fileName) {
        log.info("Tool 调用: 正在保存知识 - Q: {}, file: {}", question, fileName);

        // 1. 格式化内容
        String formattedContent = String.format("### Q：%s\n\nA：%s", question, answer);

        // 2. 处理文件名 (防止没写后缀)
        if (fileName == null || fileName.isBlank()) {
            fileName = "InfiniteChat.md"; // 默认文件
        }
        if (!fileName.endsWith(".md")) {
            fileName = fileName + ".md";
        }

        // 3. 写入物理文件
        boolean writeSuccess = appendToFile(formattedContent, fileName);
        if (!writeSuccess) {
            return "保存失败：无法写入本地文件系统，请检查日志。";
        }

        // 4. 存入向量数据库
        try {
            // 设置来源元数据
            Metadata metadata = Metadata.from("file_name", fileName);

            // 创建文档并 Embedding
            Document document = Document.from(formattedContent, metadata);
            embeddingStoreIngestor.ingest(document);

            log.info("Tool 执行成功: 知识已同步至 RAG");
            return "成功！已将该知识点保存到文档 [" + fileName + "] 并同步至向量数据库。";
        } catch (Exception e) {
            log.error("RAG - 向量化失败", e);
            return "文件写入成功，但向量数据库更新失败：" + e.getMessage();
        }
    }

    /**
     * 辅助方法：追加写入文件
     */
    private synchronized boolean appendToFile(String content, String fileName) {
        try {
            Path filePath = Paths.get(docsPath, fileName);

            // 如果文件不存在，先创建
            if (!Files.exists(filePath)) {
                if (filePath.getParent() != null) {
                    Files.createDirectories(filePath.getParent());
                }
                Files.createFile(filePath);
                log.info("Tool created new file: {}", filePath.toAbsolutePath());
            }

            // 前后加换行符
            String textToAppend = "\n\n" + content;

            Files.writeString(
                    filePath,
                    textToAppend,
                    StandardOpenOption.APPEND
            );
            return true;
        } catch (IOException e) {
            log.error("RAG Tool - 写入文件失败: {}", e.getMessage(), e);
            return false;
        }
    }
}