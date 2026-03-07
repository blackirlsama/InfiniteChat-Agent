package com.shanyangcode.infintechatagent.config;


import com.shanyangcode.infintechatagent.rag.QwenRerankClient;
import com.shanyangcode.infintechatagent.rag.ReRankingContentRetriever;
import dev.langchain4j.data.document.splitter.DocumentByParagraphSplitter;
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

    @Value("${rag.docs-path}")
    private String docsPath;

    @Bean
    public EmbeddingStoreIngestor embeddingStoreIngestor() {
        DocumentByParagraphSplitter paragraphSplitter = new DocumentByParagraphSplitter(300, 100);

        return EmbeddingStoreIngestor.builder()
                .documentSplitter(paragraphSplitter)
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

        // 第一阶段：向量召回（粗排）- 召回20条候选
        ContentRetriever baseRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(20)      // 从5改为20，增加召回量
                .minScore(0.65)      // 从0.75降至0.65，扩大召回范围
                .build();

        log.info("✅ [RAG配置] 粗排配置: maxResults=20, minScore=0.65");

        // 第二阶段：Rerank精排 - 重排后返回Top5
        ReRankingContentRetriever retriever = new ReRankingContentRetriever(baseRetriever, rerankClient, 5);

        log.info("✅ [RAG配置] Rerank精排配置: finalTopN=5");
        log.info("🎯 [RAG配置] ContentRetriever初始化完成 - 模式: 粗排(20条) → Rerank精排(Top5)");

        return retriever;
    }
}