package com.shanyangcode.infintechatagent.rag;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@Slf4j
public class ReRankingContentRetrieverTest {

    @Resource
    private ContentRetriever contentRetriever;

    @Test
    public void testRerankRetrieval() {
        // 测试查询
        String queryText = "Java多线程的实现方式有哪些？";
        Query query = Query.from(queryText);

        // 执行检索
        List<Content> results = contentRetriever.retrieve(query);

        // 验证结果
        assertNotNull(results, "检索结果不应为null");
        assertTrue(results.size() <= 5, "应返回Top5结果");
        assertTrue(results.size() > 0, "应至少返回1条结果");

        // 打印结果
        log.info("=== Rerank检索测试 ===");
        log.info("查询: {}", queryText);
        log.info("返回结果数: {}", results.size());

        for (int i = 0; i < results.size(); i++) {
            Content content = results.get(i);
            String text = content.textSegment().text();
            log.info("Top{}: {}", i + 1, text.substring(0, Math.min(100, text.length())));
        }
    }

    @Test
    public void testRerankVsBaseline() {
        String queryText = "线程安全的集合有哪些？";
        Query query = Query.from(queryText);

        List<Content> results = contentRetriever.retrieve(query);

        assertNotNull(results);
        log.info("查询: {}", queryText);
        log.info("Rerank后Top1: {}",
            results.isEmpty() ? "无结果" : results.get(0).textSegment().text().substring(0, 50));
    }
}
