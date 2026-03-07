package com.shanyangcode.infintechatagent.rag;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Component
public class QwenRerankClient {

    @Value("${langchain4j.community.dashscope.rerank-model.api-key}")
    private String apiKey;

    @Value("${langchain4j.community.dashscope.rerank-model.model-name}")
    private String modelName;

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public QwenRerankClient(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    // 通义千问Rerank API正确端点
    private static final String RERANK_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank";

    /**
     * 重排序方法
     * @param query 查询语句
     * @param documents 待排序的文档列表
     * @param topN 返回前N个结果
     * @return 排序后的文档索引列表
     */
    public List<Integer> rerank(String query, List<String> documents, int topN) {
        // 参数校验
        if (query == null || query.isEmpty()) {
            log.error("❌ Rerank查询语句不能为空");
            return null;
        }
        if (documents == null || documents.isEmpty()) {
            log.error("❌ Rerank文档列表不能为空");
            return null;
        }
        if (topN <= 0 || topN > documents.size()) {
            topN = documents.size();
            log.warn("⚠️ 调整topN为文档列表大小: {}", topN);
        }

        try {
            // 构建请求头
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);

            // 构建符合API规范的请求体
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("model", modelName);
            requestBody.put("input", Map.of(
                    "query", query,
                    "documents", documents  // 直接使用字符串数组
            ));
            requestBody.put("parameters", Map.of("top_n", topN));

            // 发送请求
            HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);
            ResponseEntity<String> responseEntity = restTemplate.postForEntity(RERANK_API_URL, requestEntity, String.class);

            // 处理响应
            if (responseEntity.getStatusCode() == HttpStatus.OK) {
                String responseBody = responseEntity.getBody();
                log.debug("✅ Rerank响应: {}", responseBody);

                // 解析响应
                Map<String, Object> responseMap = objectMapper.readValue(responseBody, Map.class);
                Map<String, Object> output = (Map<String, Object>) responseMap.get("output");
                List<Map<String, Object>> results = (List<Map<String, Object>>) output.get("results");

                // 提取排序后的索引（按relevance_score降序）
                List<Integer> rankedIndices = results.stream()
                        .sorted((r1, r2) -> Double.compare(
                                Double.parseDouble(r2.get("relevance_score").toString()),
                                Double.parseDouble(r1.get("relevance_score").toString())
                        ))
                        .map(r -> Integer.parseInt(r.get("index").toString()))
                        .collect(Collectors.toList());

                log.info("✅ [Rerank] 调用成功 - 重排序结果: {}条 -> 索引: {}",
                        documents.size(), rankedIndices);

                return rankedIndices;
            } else {
                log.error("❌ [Rerank] API调用失败 - HTTP状态码: {}, 响应: {}",
                        responseEntity.getStatusCode(),
                        responseEntity.getBody());
                return null;
            }
        } catch (Exception e) {
            log.error("❌ [Rerank] 调用异常", e);
            return null;
        }
    }
}