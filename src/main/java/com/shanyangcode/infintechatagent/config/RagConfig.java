package com.shanyangcode.infintechatagent.config;


import com.shanyangcode.infintechatagent.rag.QwenRerankClient;
import com.shanyangcode.infintechatagent.rag.QueryPreprocessor;
import com.shanyangcode.infintechatagent.rag.RecursiveDocumentSplitter;
import com.shanyangcode.infintechatagent.rag.ReRankingContentRetriever;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@Slf4j
@SuppressWarnings({"all"})
public class RagConfig {


    @Resource
    private EmbeddingModel embeddingModel;

    @Resource
    private EmbeddingStore<TextSegment> embeddingStore;

    @Resource
    private QwenRerankClient rerankClient;

    @Resource
    private QueryPreprocessor queryPreprocessor;

    @Value("${rag.docs-path}")
    private String docsPath;

    @Bean
    public EmbeddingStoreIngestor embeddingStoreIngestor() {
        log.info("🔧 创建递归文档分割器 - maxChunkSize:500, chunkOverlap:150");
        RecursiveDocumentSplitter splitter = new RecursiveDocumentSplitter(500, 150);

        return EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .textSegmentTransformer(textSegment -> TextSegment.from(
                        textSegment.metadata().getString("file_name") + "\n" + textSegment.text(),
                        textSegment.metadata()
                ))
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();
    }



    @Bean
    public ContentRetriever contentRetriever() {
        log.info("🚀 [RAG配置] 初始化ContentRetriever");

        ContentRetriever baseRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(20)
                .minScore(0.65)
                .build();

        log.info("✅ [RAG配置] 粗排配置: maxResults=20, minScore=0.65");

        ReRankingContentRetriever retriever = new ReRankingContentRetriever(
            baseRetriever,
            rerankClient,
            5,
            queryPreprocessor
        );

        log.info("✅ [RAG配置] Rerank精排配置: finalTopN=5, 查询预处理已启用");
        log.info("🎯 [RAG配置] ContentRetriever初始化完成");

        return retriever;
    }
}