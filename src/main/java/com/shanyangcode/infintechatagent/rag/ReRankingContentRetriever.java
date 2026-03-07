package com.shanyangcode.infintechatagent.rag;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@Slf4j
public class ReRankingContentRetriever implements ContentRetriever {

    private final ContentRetriever baseRetriever;
    private final QwenRerankClient rerankClient;
    private final int finalTopN;

    public ReRankingContentRetriever(ContentRetriever baseRetriever,
                                     QwenRerankClient rerankClient,
                                     int finalTopN) {
        this.baseRetriever = Objects.requireNonNull(baseRetriever, "baseRetriever不能为空");
        this.rerankClient = Objects.requireNonNull(rerankClient, "rerankClient不能为空");
        this.finalTopN = finalTopN > 0 ? finalTopN : 3; // 默认值，避免无效topN
    }

    @Override
    public List<Content> retrieve(Query query) {
        if (query == null || query.text() == null || query.text().isBlank()) {
            log.warn("⚠️ [RAG检索] 查询语句为空，直接返回空结果");
            return List.of();
        }

        log.info("📥 [RAG检索] 开始 - 查询: '{}'", query.text().substring(0, Math.min(50, query.text().length())));

        // 第一阶段：向量召回（粗排）
        log.info("🔍 [阶段1-粗排] 向量召回中...");
        List<Content> candidates = baseRetriever.retrieve(query);

        if (candidates == null || candidates.isEmpty()) {
            log.warn("⚠️ [阶段1-粗排] 向量召回无结果");
            return List.of();
        }

        log.info("✅ [阶段1-粗排] 召回 {} 条候选文档", candidates.size());

        if (candidates.size() <= finalTopN) {
            log.info("ℹ️ [跳过Rerank] 召回数量({})≤目标数量({})，直接返回", candidates.size(), finalTopN);
            // 修复：subList返回视图，转为新列表避免并发问题
            return new ArrayList<>(candidates);
        }

        // 第二阶段：Rerank精排
        log.info("🎯 [阶段2-精排] 开始Rerank重排序 ({} -> Top{})", candidates.size(), finalTopN);

        try {
            List<String> documents = candidates.stream()
                    .map(Content::textSegment)
                    .map(TextSegment::text)
                    .collect(Collectors.toList());

            List<Integer> rerankIndices = rerankClient.rerank(query.text(), documents, finalTopN);

            // 降级逻辑：Rerank失败时返回原始Top N（修复：转为新列表）
            if (rerankIndices == null || rerankIndices.isEmpty()) {
                log.warn("⚠️ [降级处理] Rerank失败，使用原始向量检索Top{}", finalTopN);
                List<Content> fallbackResults = new ArrayList<>(candidates.subList(0, Math.min(finalTopN, candidates.size())));
                log.info("📤 [RAG检索] 完成（降级模式） - 返回 {} 条结果", fallbackResults.size());
                return fallbackResults;
            }

            // 按Rerank结果重排序（核心修复：双重索引校验）
            List<Content> rerankedResults = new ArrayList<>();
            for (Integer index : rerankIndices) {
                if (index >= 0 && index < candidates.size()) { // 严格校验索引有效性
                    rerankedResults.add(candidates.get(index));
                } else {
                    log.warn("⚠️ [Rerank] 无效索引{}，跳过", index);
                }
            }

            // 兜底：如果重排序后结果为空，使用原始Top N
            if (rerankedResults.isEmpty()) {
                log.warn("⚠️ [Rerank] 重排序后无有效结果，降级为原始Top{}", finalTopN);
                rerankedResults = new ArrayList<>(candidates.subList(0, Math.min(finalTopN, candidates.size())));
            }

            log.info("✅ [阶段2-精排] Rerank完成 - 最终返回 {} 条精排结果", rerankedResults.size());
            log.info("📤 [RAG检索] 完成（Rerank模式） - 召回{}条 → 精排Top{}", candidates.size(), rerankedResults.size());

            return rerankedResults;

        } catch (Exception e) {
            log.error("❌ [异常处理] Rerank处理异常，降级为原始检索", e);
            List<Content> fallbackResults = new ArrayList<>(candidates.subList(0, Math.min(finalTopN, candidates.size())));
            log.info("📤 [RAG检索] 完成（异常降级） - 返回 {} 条结果", fallbackResults.size());
            return fallbackResults;
        }
    }
}