package com.shanyangcode.infintechatagent.job;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.*;

@Component
@Slf4j
public class RagAutoReloadJob {

    @Value("${rag.docs-path}")
    private String docsPath;

    @Resource
    private EmbeddingStoreIngestor embeddingStoreIngestor;

    private final Map<String, Long> fileTimestamps = new HashMap<>();

    @Scheduled(fixedRate = 300000) // 每5分钟检查一次
    public void autoReloadDocuments() {
        log.debug("🔄 开始扫描文档目录...");
        try {
            File dir = new File(docsPath);
            if (!dir.exists()) {
                log.warn("⚠️ 文档目录不存在: {}", docsPath);
                return;
            }

            List<Document> newDocs = new ArrayList<>();
            scanDirectory(dir, newDocs);

            if (!newDocs.isEmpty()) {
                embeddingStoreIngestor.ingest(newDocs);
                log.info("✅ 自动加载{}个新/更新文档", newDocs.size());
            } else {
                log.debug("📋 无新文档需要加载");
            }
        } catch (Exception e) {
            log.error("❌ 自动加载文档失败", e);
        }
    }

    private void scanDirectory(File dir, List<Document> newDocs) {
        File[] files = dir.listFiles();
        if (files == null) return;

        for (File file : files) {
            if (file.isDirectory()) {
                scanDirectory(file, newDocs);
            } else if (file.getName().endsWith(".md") || file.getName().endsWith(".txt")) {
                long lastModified = file.lastModified();
                Long cached = fileTimestamps.get(file.getAbsolutePath());

                if (cached == null || lastModified > cached) {
                    try {
                        Document doc = FileSystemDocumentLoader.loadDocument(file.toPath());
                        newDocs.add(doc);
                        fileTimestamps.put(file.getAbsolutePath(), lastModified);
                        log.debug("检测到新/更新文档: {}", file.getName());
                    } catch (Exception e) {
                        log.error("加载文档失败: {}", file.getName(), e);
                    }
                }
            }
        }
    }
}
