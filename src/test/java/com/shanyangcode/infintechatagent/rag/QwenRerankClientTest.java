package com.shanyangcode.infintechatagent.rag;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.Arrays;
import java.util.List;

@SpringBootTest
public class QwenRerankClientTest {

    @Autowired
    private QwenRerankClient rerankClient;

    @Test
    public void testRerank() {
        String query = "Java多线程实现方式";
        List<String> docs = Arrays.asList(
            "Java多线程可以通过继承Thread类、实现Runnable接口、实现Callable接口来实现",
            "Python的多线程使用threading模块",
            "线程池是管理线程的一种方式",
            "Java的synchronized关键字用于线程同步"
        );

        System.out.println("\n=== 测试Rerank功能 ===");
        System.out.println("查询: " + query);

        List<Integer> result = rerankClient.rerank(query, docs, 2);

        if (result != null && !result.isEmpty()) {
            System.out.println("✅ Rerank成功，Top2结果:");
            for (int i = 0; i < result.size(); i++) {
                System.out.println((i + 1) + ". " + docs.get(result.get(i)));
            }
        } else {
            System.out.println("❌ Rerank失败");
        }
    }
}
