package com.shanyangcode.infintechatagent.rag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.List;

@Component
@Slf4j
public class QueryPreprocessor {

    private static final List<String> STOP_WORDS = Arrays.asList(
        "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
        "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
        "吗", "呢", "吧", "啊", "哦", "嗯"
    );

    public String preprocess(String originalQuery) {
        if (originalQuery == null || originalQuery.length() < 3) {
            return originalQuery;
        }

        String processed = originalQuery.trim();

        // 移除标点符号
        processed = processed.replaceAll("[\\p{Punct}\\s]+", " ");

        // 移除停用词
        for (String stopWord : STOP_WORDS) {
            processed = processed.replace(stopWord, " ");
        }

        processed = processed.replaceAll("\\s+", " ").trim();

        if (!processed.equals(originalQuery)) {
            log.info("🔍 查询预处理: [{}] → [{}]", originalQuery, processed);
        }

        return processed.isEmpty() ? originalQuery : processed;
    }
}
