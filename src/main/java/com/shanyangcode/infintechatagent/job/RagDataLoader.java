package com.shanyangcode.infintechatagent.job;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.List;


/**
 * 自动加载文档
 */
@Component
@Slf4j
public class RagDataLoader implements CommandLineRunner {

    @Value("${rag.docs-path}")
    private String docsPath;

    @Resource
    private EmbeddingStoreIngestor embeddingStoreIngestor;

    @Override
    public void run(String... args) {
        log.info("=== RAG文档加载启动 ===");
        log.info("📂 文档路径: {}", docsPath);
        try {
            List<Document> documents = FileSystemDocumentLoader.loadDocuments(docsPath);

            if (!documents.isEmpty()) {
                log.info("📚 发现{}个文档，开始向量化...", documents.size());
                embeddingStoreIngestor.ingest(documents);
                log.info("✅ RAG文档加载完成 - 共处理{}个文档", documents.size());
            } else {
                log.warn("⚠️ 未发现任何文档，请检查路径: {}", docsPath);
            }
        } catch (Exception e) {
            log.error("❌ RAG文档加载失败", e);
        }
        log.info("========================");
    }
}